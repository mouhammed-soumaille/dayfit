"""The app's model (finetuned_random), defined in ONE place.

Both the experiments (evaluation/run_eval.py, method "finetuned_random") and the app (serving/engine.py)
build it with this function, so the app searches exactly like the evaluated model, and the app does not
need the evaluation code (the Docker image ships without evaluation/).
"""
SIGLIP = "hf-hub:Marqo/marqo-fashionSigLIP"
CHECKPOINT = "models/finetuned_random.pt"


def make_finetuned_random():
    from retrieval.clip_retriever import OpenClipRetriever  # torch is imported only here
    return OpenClipRetriever("finetuned_random", SIGLIP, checkpoint_path=CHECKPOINT)
