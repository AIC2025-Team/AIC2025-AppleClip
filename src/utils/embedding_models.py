import torch
import open_clip


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, _ = open_clip.create_model_and_transforms(
    'hf-hub:apple/DFN5B-CLIP-ViT-H-14', device=device
)
clip_model.eval()

def get_clip_embedding(text: str):
    with torch.no_grad():
        text_tokens = open_clip.tokenize([text]).to(device)
        text_features = clip_model.encode_text(text_tokens)
        return text_features.cpu().numpy()



