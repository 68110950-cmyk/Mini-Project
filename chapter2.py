
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

# ============================================================
# AI VISION STUDIO | CHAPTER 02
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

# เลือกฟอนต์ที่รองรับภาษาไทยบน Mac
available_fonts = {f.name for f in font_manager.fontManager.ttflist}

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
    "axes.labelcolor": WHITE,
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


def add_image_panel(fig, position, image, title, subtitle, accent,
                    cmap=None):
    """แสดงภาพในแผง Dashboard"""
    x, y, width, height = position

    add_card(
        fig,
        x,
        y,
        width,
        height,
        PANEL,
    )

    fig.text(
        x + 0.022,
        y + height - 0.045,
        title,
        fontsize=13,
        fontweight="bold",
        color=accent,
        va="center",
    )

    fig.text(
        x + 0.022,
        y + height - 0.078,
        subtitle,
        fontsize=8,
        color=MUTED,
        va="center",
    )

    ax = fig.add_axes([
        x + 0.018,
        y + 0.025,
        width - 0.036,
        height - 0.125,
    ])

    ax.set_facecolor("#080D18")
    ax.imshow(image, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)

    return ax


def main():
    print("=" * 60)
    print("AI VISION STUDIO | CHAPTER 02")
    print("การประมวลผลภาพด้วย OpenCV")
    print("OpenCV Version:", cv2.__version__)
    print("=" * 60)

    image_dir = Path(__file__).parent / "images"

    extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ) if image_dir.exists() else []

    if len(image_paths) < 3:
        raise ValueError(
            f"กรุณาใส่รูปภาพอย่างน้อย 3 รูปในโฟลเดอร์: {image_dir}"
        )

    print(f"พบรูปภาพทั้งหมด {len(image_paths)} รูป")

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"ไม่สามารถอ่านรูปภาพได้: {image_path.name}")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(
            gray_image,
            threshold1=100,
            threshold2=200,
        )

        blurred = cv2.GaussianBlur(
            rgb_image,
            (7, 7),
            0,
        )

        height, width = gray_image.shape
        channels = image.shape[2] if len(image.shape) == 3 else 1

        print(f"\n[{index}/{len(image_paths)}] {image_path.name}")
        print(f"ความละเอียด: {width} x {height}")
        print(f"จำนวน Channels: {channels}")
        print("สถานะ: ประมวลผลสำเร็จ")

        # ----------------------------------------------------
        # DASHBOARD LAYOUT
        # ----------------------------------------------------

        fig = plt.figure(figsize=(16, 11), facecolor=BG)

        # Header
        fig.text(
            0.055, 0.945,
            "AI VISION",
            fontsize=24,
            fontweight="bold",
            color=CYAN,
        )

        fig.text(
            0.055, 0.908,
            "STUDIO  /  IMAGE PROCESSING LAB",
            fontsize=10,
            color=MUTED,
        )

        fig.text(
            0.945, 0.945,
            "CHAPTER 02",
            fontsize=13,
            fontweight="bold",
            color=PURPLE,
            ha="right",
        )

        fig.text(
            0.945, 0.908,
            "OPENCV  •  COMPUTER VISION",
            fontsize=8,
            color=MUTED,
            ha="right",
        )

        # Status bar
        add_card(fig, 0.055, 0.825, 0.89, 0.055, PANEL_LIGHT)

        fig.text(
            0.075, 0.852,
            "●  SYSTEM ONLINE",
            fontsize=9,
            fontweight="bold",
            color=GREEN,
            va="center",
        )

        fig.text(
            0.35, 0.852,
            f"FILE: {image_path.name}",
            fontsize=9,
            color=WHITE,
            va="center",
        )

        fig.text(
            0.925, 0.852,
            f"{width} × {height} PX",
            fontsize=9,
            color=CYAN,
            ha="right",
            va="center",
        )

        # Four image processing panels
        panel_width = 0.425
        panel_height = 0.315

        left_x = 0.055
        right_x = 0.52
        top_y = 0.475
        bottom_y = 0.115

        add_image_panel(
            fig,
            (left_x, top_y, panel_width, panel_height),
            rgb_image,
            "01  /  ORIGINAL IMAGE",
            "ภาพต้นฉบับ • RGB COLOR",
            CYAN,
        )

        add_image_panel(
            fig,
            (right_x, top_y, panel_width, panel_height),
            gray_image,
            "02  /  GRAYSCALE",
            "แปลงภาพเป็นระดับความเข้มสีเทา",
            PURPLE,
            cmap="gray",
        )

        add_image_panel(
            fig,
            (left_x, bottom_y, panel_width, panel_height),
            edges,
            "03  /  EDGE DETECTION",
            "ตรวจจับขอบวัตถุ • CANNY",
            BLUE,
            cmap="gray",
        )

        add_image_panel(
            fig,
            (right_x, bottom_y, panel_width, panel_height),
            blurred,
            "04  /  GAUSSIAN BLUR",
            "ลดรายละเอียดรบกวนและทำให้ภาพนุ่มลง",
            GREEN,
        )

        # Footer
        fig.text(
            0.055, 0.055,
            "PROCESSING PIPELINE",
            fontsize=9,
            fontweight="bold",
            color=CYAN,
        )

        fig.text(
            0.055, 0.032,
            "Original  →  Grayscale  →  Canny  →  Gaussian Blur",
            fontsize=9,
            color=MUTED,
        )

        fig.text(
            0.945, 0.04,
            f"IMAGE {index:02d}  /  {len(image_paths):02d}",
            fontsize=9,
            color=PURPLE,
            ha="right",
        )

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
