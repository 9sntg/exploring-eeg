# clip_embedding.py
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


def compute_image_embeddings(model, processor, image_paths, batch_size, device):
    model.eval()
    all_embs = []

    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), batch_size), desc="Image embeddings"):
            batch = image_paths[start:start + batch_size]
            imgs = []
            for p in batch:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except:
                    imgs.append(Image.new("RGB", (224, 224)))

            inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
            feats = model.get_image_features(**inputs)
            feats = F.normalize(feats, p=2, dim=1)
            all_embs.append(feats.cpu())

    return torch.cat(all_embs, dim=0)


def compute_text_embeddings(model, processor, texts, batch_size, device):
    model.eval()
    all_embs = []

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Text embeddings"):
            batch = texts[start:start + batch_size]

            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
            feats = model.get_text_features(**inputs)
            feats = F.normalize(feats, p=2, dim=1)
            all_embs.append(feats.cpu())

    return torch.cat(all_embs, dim=0)


def batch_retrieve_captions(image_embs, text_embs, k=5):
    sim = image_embs @ text_embs.T
    top_scores, top_idx = torch.topk(sim, k=k, dim=1)
    return top_idx.cpu().numpy(), top_scores.cpu().numpy(), sim