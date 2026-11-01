
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


def main():
    device = torch.device("cpu")

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    print("=" * 55)
    print("Chapter 1: ResNet18 Image Classification")
    print("Chapter 2: OpenCV Image Processing")
    print("=" * 55)
    print("Device:", device)
    print("Categories:", len(categories))
    print("OpenCV:", cv2.__version__)
    print("NumPy:", np.__version__)

    image_dir = Path(__file__).parent / "images"
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if image_dir.exists()
        and p.is_file()
        and p.suffix.lower() in extensions
    ) if image_dir.exists() else []

    if len(image_paths) < 3:
        raise ValueError(
            f"Please put at least 3 images in: {image_dir}"
        )

    print(f"\nFound {len(image_paths)} images.")

    for image_path in image_paths:
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")

            tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.inference_mode():
                logits = model(tensor)
                probabilities = torch.softmax(logits[0], dim=0)
                scores, indices = torch.topk(probabilities, 5)

            results = [
                (categories[int(i)], float(s))
                for s, i in zip(scores, indices)
            ]

            print(f"\nImage: {image_path.name}")
            for rank, (label, score) in enumerate(results, 1):
                print(f"{rank}. {label}: {score:.2%}")

            # Chapter 2: OpenCV image processing
            cv_image = cv2.imread(str(image_path))

            if cv_image is None:
                print("OpenCV could not read this image.")
                continue

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            labels = [label for label, _ in results]
            values = [score for _, score in results]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(gray, cmap="gray")
            axes[1].set_title("Grayscale (OpenCV)")
            axes[1].axis("off")

            axes[2].barh(
                range(5), values, color="#1677d2"
            )
            axes[2].set_yticks(range(5), labels=labels)
            axes[2].invert_yaxis()
            axes[2].set_xlim(0, 1)
            axes[2].set_title("ResNet18 Top-5")
            axes[2].set_xlabel("Softmax score")

            for i, score in enumerate(values):
                axes[2].text(
                    min(score + 0.02, 0.82),
                    i,
                    f"{score:.1%}",
                    va="center",
                )

            plt.tight_layout()
            plt.show()

            # Display edge detection result separately
            plt.figure(figsize=(7, 5))
            plt.imshow(edges, cmap="gray")
            plt.title(f"OpenCV Edge Detection: {image_path.name}")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        except Exception as error:
            print(f"\nError processing {image_path.name}: {error}")


if __name__ == "__main__":
    main()
