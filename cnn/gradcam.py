import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from engine import PetClassifier
from loader import get_dataloaders


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class_idx):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[:, target_class_idx].sum()
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
        return cam.squeeze().detach().cpu().numpy(), logits.squeeze().detach().cpu()

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()


def overlay_cam(img_np, cam, alpha=0.5):
    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam)[:, :, :3]
    overlay = (1 - alpha) * img_np + alpha * heatmap
    return np.clip(overlay, 0, 1), heatmap


def process_and_save(grad_cam, image_tensor, original_img, class_idx, class_name, output_path, device):
    input_tensor = image_tensor.unsqueeze(0).to(device)
    cam, logits = grad_cam.generate(input_tensor, class_idx)
    prob = torch.sigmoid(logits)[class_idx].item()
    logit_val = logits[class_idx].item()

    img_resized = original_img.resize((224, 224))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    overlay, heatmap = overlay_cam(img_np, cam)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"{class_name} (Prob: {prob:.3f}, Logit: {logit_val:.2f})")
    axes[2].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def run_gradcam(checkpoint_path, mode, output_dir, image_path=None, num_samples=5, target_class=None):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    labels = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]

    model = PetClassifier.load_from_checkpoint(checkpoint_path, mode=mode, strict=False)
    model.eval()
    model.to(device)

    target_layer = model.attention
    grad_cam = GradCAM(model, target_layer)

    out_dir = Path(output_dir) / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if image_path:
        img_path = Path(image_path)
        with Image.open(img_path) as img:
            original_img = img.convert("RGB")
        image_tensor = transform(original_img)

        with torch.no_grad():
            preds = model(image_tensor.unsqueeze(0).to(device))
            top_class = target_class if target_class else labels[preds.argmax().item()]

        target_idx = labels.index(top_class)
        save_path = out_dir / f"{img_path.stem}_{top_class}.png"
        process_and_save(grad_cam, image_tensor, original_img, target_idx, top_class, save_path, device)
    else:
        _, val_loader, _ = get_dataloaders(fold_idx=0, batch_size=1, num_workers=0)
        val_df = val_loader.dataset.df

        for idx in range(min(num_samples, len(val_df))):
            row = val_df.iloc[idx]
            image_id = row["image_id"]
            img_path = val_loader.dataset.img_dir / f"{image_id}.jpg"
            if not img_path.exists():
                continue

            with Image.open(img_path) as img:
                original_img = img.convert("RGB")
            image_tensor = transform(original_img)

            if target_class:
                target_classes = [target_class]
            else:
                active_classes = [lbl for lbl in labels if row[lbl] == 1]
                target_classes = active_classes if active_classes else [labels[0]]

            for cls_name in target_classes:
                cls_idx = labels.index(cls_name)
                save_path = out_dir / f"{image_id}_{cls_name}.png"
                process_and_save(grad_cam, image_tensor, original_img, cls_idx, cls_name, save_path, device)

    grad_cam.remove_hooks()


def get_args():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument("--checkpoint", "--ckpt", type=str, required=True)
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "se", "cbam"])
    parser.add_argument("--output_dir", type=str, default="reports/gradcam")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--target_class", type=str, default=None, choices=["Cat", "Dog", "Moris", "Motya", "Biatrix"])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_gradcam(
        checkpoint_path=args.checkpoint,
        mode=args.mode,
        output_dir=args.output_dir,
        image_path=args.image_path,
        num_samples=args.num_samples,
        target_class=args.target_class
    )
