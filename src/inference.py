import argparse
import torch
from torchvision import transforms
from PIL import Image
import json
from src.defornet import build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    return ckpt

def build_inference_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    img_size = ckpt["img_size"]
    channels = ckpt.get("channels", [32,64,128,256])

    model, _ = build_model(num_classes=len(class_to_idx), img_size=img_size, channels=channels)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    tfms = build_inference_transforms(img_size)
    img = Image.open(args.image).convert("RGB")
    tensor = tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    print("Top predictions:")
    for idx, p in ranked[:5]:
        print(f"{idx_to_class[idx]}: {p:.4f}")

    print("Full prob JSON:")
    print(json.dumps({idx_to_class[i]: float(p) for i, p in enumerate(probs)}, indent=2))

if __name__ == "__main__":
    main()