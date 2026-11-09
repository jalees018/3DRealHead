import clip
import torch
import torch.nn.functional as F
from PIL import Image


@torch.no_grad()
def clip_identity_embeddings(image_paths, device, batch_size, style_dim):
    """CLIP (ViT-B/32) image embedding per subject, L2-normalised and repeated over the batch.

    Returns a list with one (batch_size, style_dim) tensor per image.
    """
    model, preprocess = clip.load("ViT-B/32", device=device)
    embeddings = []
    for path in image_paths:
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        features = model.encode_image(image)
        embeddings.append(F.normalize(features, dim=1).float().repeat((batch_size, 1))[:, 0:style_dim])
    return embeddings
