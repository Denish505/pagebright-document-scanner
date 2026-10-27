"""Train a page-corner regressor from PageBright's exported label ZIPs."""
from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class CornerDataset(Dataset):
    def __init__(self, root: Path):
        self.samples: list[tuple[bytes, np.ndarray]] = []
        for archive_path in sorted(root.glob("*.zip")):
            with zipfile.ZipFile(archive_path) as archive:
                labels = json.loads(archive.read("labels.json"))
                image_bytes = archive.read(labels.get("image", "page.jpg"))
            self.samples.append(
                (image_bytes, np.asarray(labels["corners_normalized"], dtype=np.float32).reshape(8))
            )
        for label_path in sorted(root.rglob("labels.json")):
            labels = json.loads(label_path.read_text(encoding="utf-8"))
            image_path = label_path.parent / labels.get("image", "page.jpg")
            if image_path.is_file():
                self.samples.append(
                    (image_path.read_bytes(), np.asarray(labels["corners_normalized"], dtype=np.float32).reshape(8))
                )
        if len(self.samples) < 30:
            raise ValueError(f"Need at least 30 labeled samples; found {len(self.samples)}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_bytes, corners = self.samples[index]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        tensor = transforms.Normalize(MEAN, STD)(transforms.ToTensor()(image))
        return tensor, torch.from_numpy(corners.copy())


class CornerRegressor(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(features, 8)

    def forward(self, tensor):
        return torch.sigmoid(self.backbone(tensor))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing labeled ZIPs or extracted samples.")
    parser.add_argument("--output", type=Path, default=Path("models/pagebright_corners.onnx"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--no-pretrained", action="store_true", help="Train without downloading initialization weights.")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = CornerDataset(args.data_dir)
    validation_size = max(1, int(round(len(dataset) * 0.2)))
    training_size = len(dataset) - validation_size
    training_data, validation_data = random_split(
        dataset, [training_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CornerRegressor(pretrained=not args.no_pretrained)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.SmoothL1Loss()
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, corners in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), corners)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * images.shape[0]
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for images, corners in validation_loader:
                validation_loss += float(loss_fn(model(images), corners)) * images.shape[0]
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs}: "
            f"train={train_loss / training_size:.5f}, "
            f"validation={validation_loss / validation_size:.5f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    torch.onnx.export(
        model, example, args.output,
        input_names=["image"], output_names=["corners"], opset_version=17, dynamo=False,
    )
    torch.save(model.state_dict(), args.output.with_suffix(".pt"))
    print(f"Saved ONNX model: {args.output}")
    print(f"Saved checkpoint: {args.output.with_suffix('.pt')}")
    print("Place the ONNX file in the app's models directory to enable learned detection.")


if __name__ == "__main__":
    main()
