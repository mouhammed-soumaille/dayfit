"""Fine-tune FashionSigLIP's image encoder for phone-photo -> shop-photo search.

Training data (no extra downloads needed):
  For every catalog item that is NOT a test query, we make a fresh synthetic
  phone photo of it (data_prep/phone_aug.py, random severity, a new random
  version every epoch) and pair it with the clean shop photo.
  -> test items are never trained on; they only appear as gallery distractors.

Loss (contrastive, "in-batch negatives"):
  A batch has B pairs. We embed the B phone photos and the B shop photos, and
  compute all B x B cosine similarities. For phone photo i the right answer is
  shop photo i; the other B-1 shop photos in the batch are the wrong answers.
  Cross-entropy pushes the right pair's similarity up and the wrong ones down
  (in both directions: phone->shop and shop->phone).

Hard negatives (--hard_batches, on by default):
  Each batch is filled with items of the SAME article type, so the wrong
  answers are similar garments (other T-shirts, not shoes). Telling apart
  similar items is exactly what search needs, so these teach the model more.

Keeping it small enough for a laptop:
  only the last --train_blocks transformer blocks (and the final pooling /
  projection layers) are trained; everything else stays frozen.

Validation:
  ~300 held-out non-test items, one medium phone photo each (fixed seed),
  searched against the whole catalog minus the test items. Recall@1 is
  measured before training (= the pretrained model) and after every epoch;
  the best epoch is saved to models/finetuned.pt (Contract 4).

Run from the repo root:
    python -m training.finetune --smoke      # 1-2 minute check that everything works
    python -m training.finetune              # the real run
"""
import argparse
import csv
import os
import time

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data_prep.phone_aug import make_phone_version
from retrieval.clip_retriever import pick_device

SEVERITIES = ["mild", "medium", "strong"]


# ---------------------------------------------------------------- data

def split_items(catalog, queries, n_val, seed):
    """Remove every test-query item, then split the rest into train / val."""
    test_ids = set(queries["source_item_id"].dropna().astype(str))
    pool = catalog[~catalog["item_id"].isin(test_ids)].sample(frac=1, random_state=seed)
    val = pool.iloc[:n_val].reset_index(drop=True)
    train = pool.iloc[n_val:].reset_index(drop=True)
    gallery = catalog[~catalog["item_id"].isin(test_ids)].reset_index(drop=True)
    return train, val, gallery


class PairDataset(Dataset):
    """Item i -> (synthetic phone photo, clean shop photo), both as model input tensors."""

    def __init__(self, df, preprocess, seed):
        self.paths = list(df["image_path"])
        self.preprocess = preprocess
        self.seed = seed
        self.epoch = 0  # changed every epoch, so each epoch sees new phone versions

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        clean = Image.open(self.paths[i]).convert("RGB")
        rng = np.random.default_rng([self.seed, self.epoch, i])
        severity = SEVERITIES[rng.integers(len(SEVERITIES))]
        phone = make_phone_version(clean, rng, severity)
        return self.preprocess(phone), self.preprocess(clean)


def make_batches(types, batch_size, rng, hard):
    """List of batches (lists of item indices) for one epoch.
    hard=True: every batch holds items of a single article type."""
    if not hard:
        order = rng.permutation(len(types)).tolist()
        batches = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
    else:
        by_type = {}
        for i, t in enumerate(types):
            by_type.setdefault(t, []).append(i)
        batches = []
        for indices in by_type.values():
            indices = rng.permutation(indices).tolist()
            batches += [indices[s:s + batch_size] for s in range(0, len(indices), batch_size)]
        rng.shuffle(batches)
    return [b for b in batches if len(b) > 1]  # a batch of 1 has no negatives


# ---------------------------------------------------------------- model

def find_blocks(visual):
    """The list of transformer blocks inside the image encoder."""
    if hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
        return list(visual.trunk.blocks)            # timm ViT (SigLIP models)
    if hasattr(visual, "transformer"):
        return list(visual.transformer.resblocks)   # open_clip ViT (CLIP models)
    raise ValueError("don't know where this model's transformer blocks are")


def set_trainable(model, n_blocks):
    """Freeze everything, then unfreeze the last n_blocks blocks of the image
    encoder plus the layers after them (final norm, pooling, projection)."""
    for p in model.parameters():
        p.requires_grad = False
    blocks = find_blocks(model.visual)
    for block in blocks[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True
    for name, module in model.visual.named_modules():
        if name.split(".")[-1] in ("norm", "fc_norm", "attn_pool", "head", "ln_post"):
            for p in module.parameters():
                p.requires_grad = True
    if isinstance(getattr(model.visual, "proj", None), torch.nn.Parameter):
        model.visual.proj.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_visual = sum(p.numel() for p in model.visual.parameters())
    print(f"training {n_train / 1e6:.1f}M of the image encoder's {n_visual / 1e6:.1f}M parameters "
          f"(last {n_blocks} of {len(blocks)} blocks)")


@torch.no_grad()
def embed(model, preprocess, images, device, batch_size=64):
    """images: list of PIL images or file paths -> (n, d) normalised embeddings on CPU."""
    out = []
    for s in range(0, len(images), batch_size):
        chunk = [Image.open(x).convert("RGB") if isinstance(x, str) else x
                 for x in images[s:s + batch_size]]
        batch = torch.stack([preprocess(img) for img in chunk]).to(device)
        out.append(F.normalize(model.encode_image(batch), dim=-1).float().cpu())
    return torch.cat(out)


def validate(model, preprocess, gallery, val, val_phone, device):
    """Recall@1 and Recall@5 of the val phone photos against the gallery."""
    gallery_emb = embed(model, preprocess, list(gallery["image_path"]), device)
    query_emb = embed(model, preprocess, val_phone, device)
    scores = query_emb @ gallery_emb.T
    position = {item_id: i for i, item_id in enumerate(gallery["item_id"])}
    targets = torch.tensor([position[i] for i in val["item_id"]])
    top5 = scores.topk(5, dim=1).indices
    r1 = (top5[:, 0] == targets).float().mean().item()
    r5 = (top5 == targets[:, None]).any(dim=1).float().mean().item()
    return r1, r5


def contrastive_loss(phone_emb, shop_emb, temperature):
    """Symmetric cross-entropy over the B x B similarity matrix (right answer = diagonal)."""
    logits = phone_emb @ shop_emb.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hf-hub:Marqo/marqo-fashionSigLIP")
    parser.add_argument("--catalog", default="data/catalog.csv")
    parser.add_argument("--queries", default="data/queries.csv")
    parser.add_argument("--out", default="models/finetuned.pt")
    parser.add_argument("--log", default="models/finetune_log.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--train_blocks", type=int, default=4)
    parser.add_argument("--n_val", type=int, default=300)
    parser.add_argument("--hard_batches", type=int, default=1, help="1 = same-type batches, 0 = random")
    parser.add_argument("--workers", type=int, default=0, help="data-loading processes (0 = main process)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the code works")
    args = parser.parse_args()

    if args.smoke:
        args.epochs, args.batch_size, args.n_val = 1, 8, 20
        args.out, args.log = "models/finetuned_smoke.pt", "models/finetune_smoke_log.csv"

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device()

    catalog = pd.read_csv(args.catalog, dtype={"item_id": str})
    queries = pd.read_csv(args.queries, dtype={"source_item_id": str})
    train, val, gallery = split_items(catalog, queries, args.n_val, args.seed)
    if args.smoke:
        train = train.iloc[:48]
        gallery = pd.concat([val, gallery.iloc[:200]]).drop_duplicates("item_id").reset_index(drop=True)
    print(f"train items: {len(train)}   val items: {len(val)}   gallery: {len(gallery)}   device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(args.model)
    model = model.to(device)
    model.eval()  # no dropout / random layers; gradients still flow into the trainable blocks
    set_trainable(model, args.train_blocks)

    # fixed validation photos: the same medium-severity phone photo every epoch
    val_phone = [make_phone_version(Image.open(p).convert("RGB"),
                                    np.random.default_rng([args.seed, 999, i]), "medium")
                 for i, p in enumerate(val["image_path"])]

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    dataset = PairDataset(train, preprocess, args.seed)
    steps_per_epoch = len(make_batches(list(train["article_type"]), args.batch_size,
                                       np.random.default_rng(0), args.hard_batches))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    log_file = open(args.log, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["epoch", "train_loss", "val_recall@1", "val_recall@5", "minutes"])

    start = time.time()
    best_r1, _ = validate(model, preprocess, gallery, val, val_phone, device)
    print(f"epoch 0 (pretrained): val Recall@1 = {best_r1:.3f}")
    log.writerow([0, "", f"{best_r1:.4f}", "", f"{(time.time() - start) / 60:.1f}"])
    saved_any = False

    for epoch in range(1, args.epochs + 1):
        dataset.epoch = epoch
        batches = make_batches(list(train["article_type"]), args.batch_size, rng, args.hard_batches)
        loader = DataLoader(dataset, batch_sampler=batches, num_workers=args.workers)

        losses = []
        for step, (phone, shop) in enumerate(loader, start=1):
            phone, shop = phone.to(device), shop.to(device)
            phone_emb = F.normalize(model.encode_image(phone), dim=-1)
            shop_emb = F.normalize(model.encode_image(shop), dim=-1)
            loss = contrastive_loss(phone_emb, shop_emb, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            print(f"  epoch {epoch}  step {step}/{len(batches)}  loss {np.mean(losses[-20:]):.3f}", end="\r")

        r1, r5 = validate(model, preprocess, gallery, val, val_phone, device)
        minutes = (time.time() - start) / 60
        print(f"\nepoch {epoch}: train loss {np.mean(losses):.3f}   val Recall@1 {r1:.3f}   "
              f"Recall@5 {r5:.3f}   ({minutes:.1f} min so far)")
        log.writerow([epoch, f"{np.mean(losses):.4f}", f"{r1:.4f}", f"{r5:.4f}", f"{minutes:.1f}"])
        log_file.flush()

        if r1 > best_r1:
            best_r1 = r1
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"state_dict": state, "base_model": args.model, "epoch": epoch,
                        "val_recall@1": r1}, args.out)
            saved_any = True
            print(f"  -> best so far, saved {args.out}")

    log_file.close()
    if not saved_any:
        print("No epoch beat the pretrained model on validation, so no checkpoint was saved. "
              "That is a valid result to report.")
    print(f"done. log: {args.log}")


if __name__ == "__main__":
    main()
