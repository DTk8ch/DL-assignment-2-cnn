import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
import datetime

from engine import PetClassifier


class CatsDogsTestDataset(Dataset):
    """
    A modified dataset class exclusively for inference.
    It drops the target extraction that would crash on a test.csv.
    """

    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['image_id']}.jpg"

        with Image.open(img_path) as img:
            image = img.convert("RGB")
            if self.transform:
                image = self.transform(image)

        return image, row['image_id']


def run_inference(ckpt_path, test_csv_path, img_dir, output_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")

    print(f"Loading checkpoint: {ckpt_path}")
    model = PetClassifier.load_from_checkpoint(ckpt_path, strict=False)
    model.eval()
    model.to(device)

    df_test = pd.read_csv(test_csv_path)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = CatsDogsTestDataset(df_test, img_dir, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    image_ids = []
    all_predictions = []

    print("Running forward passes...")

    optimal_thresholds = torch.tensor([0.75, 0.29, 0.89, 0.99, 0.97], device=device)

    with torch.no_grad():
        for images, batch_img_ids in test_loader:
            images = images.to(device)

            logits = model(images)

            probs = torch.sigmoid(logits)

            probs_interpreted = (probs >= optimal_thresholds).int()

            image_ids.extend(batch_img_ids)
            all_predictions.extend(probs_interpreted.cpu().numpy())

    print("Formatting submission...")
    labels = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]

    submission_df = pd.DataFrame(all_predictions, columns=labels)
    submission_df.insert(0, 'image_id', image_ids)

    submission_df.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")


def get_args():
    parser = argparse.ArgumentParser(description='Specify the path to the model to be submitted')

    parser.add_argument(
        '--path', '--p',
        type=str,
        required=True
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    CKPT_FILE = args.path

    TEST_CSV = "data/test.csv"
    TEST_IMG_DIR = "data/test"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    SUBMISSION_FILE = f"submissions/submission_optimized_{timestamp}.csv"

    run_inference(CKPT_FILE, TEST_CSV, TEST_IMG_DIR, SUBMISSION_FILE)
