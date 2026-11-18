"""Why does a photo get the results it gets? (a debugging tool for the app, not an evaluation)

For one photo it prints the top 8 of the app catalogue for three versions of the photo, and saves
what the model actually looks at for each version (my_photos/<name>_seen_<version>.png):
  as_is   : exactly what the API does (EXIF rotation, then RGB). A transparent background becomes
            whatever colour is hidden under the transparency, often black.
  on_white: transparent pixels put on white first.
  padded  : on_white, then placed in the middle of a white square.
Why "padded": the model's preprocessing resizes the SHORT side to 224 pixels and keeps only the
centre square, so for a tall photo the top and bottom are cut off (a long pair of jeans can lose its
waistband and hems). Padding to a square first keeps the whole garment in view.

Sanity check: a catalogue image used as the query finds itself first with similarity ~1.

Run from the repo root (loads the model, about 20 seconds):
    python -m serving.diagnose_photo my_photos/jeans.jpg
"""
import argparse
import os

from PIL import Image, ImageOps

from retrieval.preprocess import pad_to_square
from serving.engine import SearchEngine


def versions(path):
    image = ImageOps.exif_transpose(Image.open(path))
    print(f"  file: mode {image.mode}, size {image.size[0]} x {image.size[1]} (width x height)")
    rgba = image.convert("RGBA")
    on_white = Image.new("RGB", rgba.size, (255, 255, 255))
    on_white.paste(rgba, mask=rgba.getchannel("A"))          # transparent -> white
    return {"as_is": image.convert("RGB"), "on_white": on_white, "padded": pad_to_square(on_white)}


def what_the_model_sees(engine, image):
    """The 224 x 224 picture the model receives, turned back into an image."""
    steps = engine.retriever.preprocess.transforms
    norm = [t for t in steps if type(t).__name__ == "Normalize"][0]
    x = engine.retriever.preprocess(image)                   # 3 x 224 x 224, normalised
    for c in range(3):                                       # undo the normalisation
        x[c] = x[c] * norm.std[c] + norm.mean[c]
    pixels = (x.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(pixels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    args = parser.parse_args()

    engine = SearchEngine(catalogue="app")
    first = engine.catalog.iloc[0]
    found, _ = engine.search(Image.open(first["image_path"]), k=1)
    print(f"  check: catalogue item {first['item_id']} as query -> top 1 {found[0]['item_id']}, "
          f"similarity {found[0]['score']:.3f}")
    if found[0]["item_id"] != first["item_id"]:
        raise SystemExit("sanity check failed")

    name = os.path.splitext(os.path.basename(args.photo))[0]
    os.makedirs("my_photos", exist_ok=True)
    for version, image in versions(args.photo).items():
        what_the_model_sees(engine, image).save(f"my_photos/{name}_seen_{version}.png")
        found, _ = engine.search(image, k=8)
        print(f"\n  {version}:")
        for f in found:
            source = engine.item.loc[f["item_id"], "source"]
            print(f"    {f['rank']}. {f['score']:.3f}  {f['article_type']:<14} {str(f['color']):<10} "
                  f"{source:<10} {f['item_id']}")
    print(f"\n  saved what the model sees: my_photos/{name}_seen_*.png")


if __name__ == "__main__":
    main()
