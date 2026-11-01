
# บทที่ 1: การจำแนกรูปภาพด้วยโมเดล ResNet18 ที่ผ่านการฝึกมาแล้ว

from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


# ============================================================
# AI VISION STUDIO | CHAPTER 01
# ============================================================

BG = "#080D1A"
PANEL = "#111A2D"
PANEL_LIGHT = "#17243B"
CYAN = "#70D6FF"
PURPLE = "#B794F6"
BLUE = "#638CFF"
GREEN = "#55E6A5"
WHITE = "#F4F7FF"
MUTED = "#91A0BA"
BORDER = "#293854"

available_fonts = {
    font.name for font in font_manager.fontManager.ttflist
}

thai_fonts = [
    "Thonburi",
    "Tahoma",
    "Leelawadee UI",
    "Noto Sans Thai",
    "Arial Unicode MS",
]

FONT = next(
    (name for name in thai_fonts if name in available_fonts),
    "DejaVu Sans",
)

plt.rcParams.update({
    "font.family": FONT,
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER,
    "text.color": WHITE,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


def add_card(fig, x, y, width, height, color=PANEL):
    """สร้างการ์ดพื้นหลังมุมโค้ง"""

    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        transform=fig.transFigure,
        facecolor=color,
        edgecolor=BORDER,
        linewidth=1.0,
        zorder=-1,
    )

    fig.patches.append(card)


def get_device():
    """เลือกอุปกรณ์ที่รองรับโดยอัตโนมัติ"""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def show_dashboard(
    image,
    image_path,
    results,
    device_name,
    image_number,
    total_images,
):
    """แสดงผลการจำแนกภาพใน AI Dashboard"""

    fig = plt.figure(figsize=(16, 10), facecolor=BG)

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    fig.text(
        0.055, 0.945,
        "AI VISION",
        fontsize=25,
        fontweight="bold",
        color=CYAN,
    )

    fig.text(
        0.055, 0.908,
        "STUDIO  /  IMAGE CLASSIFICATION",
        fontsize=10,
        color=MUTED,
    )

    fig.text(
        0.945, 0.945,
        "CHAPTER 01",
        fontsize=13,
        fontweight="bold",
        color=PURPLE,
        ha="right",
    )

    fig.text(
        0.945, 0.908,
        "DEEP LEARNING  •  RESNET18",
        fontsize=9,
        color=MUTED,
        ha="right",
    )

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    add_card(fig, 0.055, 0.825, 0.89, 0.055, PANEL_LIGHT)

    fig.text(
        0.075, 0.852,
        "●  MODEL READY",
        fontsize=9,
        fontweight="bold",
        color=GREEN,
        va="center",
    )

    fig.text(
        0.36, 0.852,
        f"DEVICE: {device_name.upper()}",
        fontsize=9,
        color=CYAN,
        va="center",
    )

    fig.text(
        0.925, 0.852,
        "IMAGENET-1K  /  1000 CLASSES",
        fontsize=9,
        color=PURPLE,
        ha="right",
        va="center",
    )

    # --------------------------------------------------------
    # LEFT PANEL: ORIGINAL IMAGE
    # --------------------------------------------------------

    add_card(fig, 0.055, 0.205, 0.405, 0.585)

    fig.text(
        0.08, 0.748,
        "01  /  INPUT IMAGE",
        fontsize=13,
        fontweight="bold",
        color=CYAN,
    )

    fig.text(
        0.08, 0.713,
        image_path.name,
        fontsize=9,
        color=MUTED,
    )

    image_ax = fig.add_axes([0.08, 0.345, 0.355, 0.335])

    image_ax.imshow(image)
    image_ax.set_xticks([])
    image_ax.set_yticks([])

    for spine in image_ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(1.2)

    width, height = image.size

    fig.text(
        0.08, 0.285,
        "IMAGE RESOLUTION",
        fontsize=8,
        color=MUTED,
    )

    fig.text(
        0.08, 0.247,
        f"{width} × {height} PX",
        fontsize=15,
        fontweight="bold",
        color=WHITE,
    )

    fig.text(
        0.435, 0.247,
        "RGB",
        fontsize=10,
        fontweight="bold",
        color=GREEN,
        ha="right",
    )

    # --------------------------------------------------------
    # RIGHT PANEL: TOP-5 PREDICTIONS
    # --------------------------------------------------------

    add_card(fig, 0.49, 0.205, 0.455, 0.585)

    fig.text(
        0.515, 0.748,
        "02  /  PREDICTION RESULTS",
        fontsize=13,
        fontweight="bold",
        color=PURPLE,
    )

    fig.text(
        0.515, 0.713,
        "TOP-5 IMAGE CLASSIFICATION",
        fontsize=9,
        color=MUTED,
    )

    # ผลลัพธ์อันดับ 1
    best_label, best_score = results[0]

    add_card(
        fig,
        0.515, 0.595, 0.405, 0.085,
        PANEL_LIGHT,
    )

    fig.text(
        0.535, 0.645,
        "BEST MATCH",
        fontsize=8,
        color=GREEN,
        fontweight="bold",
    )

    fig.text(
        0.535, 0.612,
        best_label,
        fontsize=12,
        color=WHITE,
        fontweight="bold",
    )

    fig.text(
        0.9, 0.63,
        f"{best_score:.1%}",
        fontsize=17,
        color=CYAN,
        fontweight="bold",
        ha="right",
    )

    # กราฟ Top 5
    bar_ax = fig.add_axes([0.53, 0.29, 0.375, 0.265])

    labels = [label for label, _ in results]
    scores = [score for _, score in results]
    positions = list(range(len(results)))

    bar_colors = [
        CYAN,
        BLUE,
        PURPLE,
        "#596783",
        "#46536D",
    ]

    bars = bar_ax.barh(
        positions,
        scores,
        height=0.55,
        color=bar_colors,
        edgecolor="none",
        zorder=3,
    )

    bar_ax.set_yticks(positions)
    bar_ax.set_yticklabels(labels, fontsize=9)
    bar_ax.invert_yaxis()
    bar_ax.set_xlim(0, 1.12)

    bar_ax.set_xlabel(
        "SOFTMAX SCORE",
        fontsize=8,
        labelpad=8,
    )

    bar_ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    bar_ax.set_xticklabels(
        ["0%", "25%", "50%", "75%", "100%"],
        fontsize=8,
    )

    bar_ax.grid(
        axis="x",
        color=BORDER,
        linestyle="--",
        linewidth=0.7,
        alpha=0.8,
        zorder=0,
    )

    bar_ax.set_axisbelow(True)

    for spine in bar_ax.spines.values():
        spine.set_visible(False)

    bar_ax.tick_params(
        axis="y",
        length=0,
        pad=8,
    )

    bar_ax.tick_params(
        axis="x",
        length=0,
        pad=6,
    )

    for bar, score in zip(bars, scores):
        bar_ax.text(
            min(score + 0.025, 1.025),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1%}",
            va="center",
            ha="left",
            fontsize=9,
            color=WHITE,
            fontweight="bold",
            clip_on=False,
        )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    fig.text(
        0.055, 0.12,
        "MODEL ARCHITECTURE",
        fontsize=8,
        color=MUTED,
    )

    fig.text(
        0.055, 0.085,
        "ResNet18",
        fontsize=12,
        color=CYAN,
        fontweight="bold",
    )

    fig.text(
        0.30, 0.12,
        "INFERENCE MODE",
        fontsize=8,
        color=MUTED,
    )

    fig.text(
        0.30, 0.085,
        "torch.inference_mode()",
        fontsize=10,
        color=PURPLE,
    )

    fig.text(
        0.945, 0.105,
        f"IMAGE {image_number:02d} / {total_images:02d}",
        fontsize=9,
        color=GREEN,
        ha="right",
    )

    fig.text(
        0.945, 0.075,
        "PYTORCH  •  TORCHVISION  •  MATPLOTLIB",
        fontsize=8,
        color=MUTED,
        ha="right",
    )

    plt.show()
    plt.close(fig)


def main():
    print("=" * 60)
    print("AI VISION STUDIO | CHAPTER 01")
    print("ResNet18 Image Classification")
    print("=" * 60)

    device = get_device()

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    print("Device:", device)
    print("Model: ResNet18")
    print("Training dataset: ImageNet-1K")
    print("Number of categories:", len(categories))

    image_dir = Path(__file__).parent / "images"

    extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
    ) if image_dir.exists() else []

    if len(image_paths) < 3:
        raise ValueError(
            f"Please put at least 3 images in: {image_dir}"
        )

    print(f"\nFound {len(image_paths)} images.")

    successful_images = 0

    for image_path in image_paths:
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")

            input_tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.inference_mode():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits[0], dim=0)

                top_scores, top_indices = torch.topk(
                    probabilities,
                    k=5,
                )

            results = [
                (
                    categories[int(index)],
                    float(score),
                )
                for score, index in zip(
                    top_scores.cpu(),
                    top_indices.cpu(),
                )
            ]

            print(f"\nImage: {image_path.name}")
            print("-" * 60)

            for rank, (label, score) in enumerate(
                results,
                start=1,
            ):
                print(
                    f"{rank}. {label:30s} "
                    f"{score:.4f} ({score:.1%})"
                )

            successful_images += 1

            show_dashboard(
                image=image,
                image_path=image_path,
                results=results,
                device_name=str(device),
                image_number=successful_images,
                total_images=len(image_paths),
            )

        except Exception as error:
            print(
                f"\nError processing {image_path.name}: {error}"
            )

    print("\n" + "=" * 60)
    print("IMAGE CLASSIFICATION COMPLETED")
    print(f"Successfully processed: {successful_images}")
    print("=" * 60)


if __name__ == "__main__":
    main()
