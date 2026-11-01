
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    # ตั้งค่ารูปแบบกราฟให้ดูสะอาดและทันสมัย
    plt.rcParams.update({
        "figure.facecolor": "#FFF8FC",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#E5DCE5",
        "axes.labelcolor": "#424052",
        "text.color": "#424052",
        "xtick.color": "#666477",
        "ytick.color": "#666477",
        "font.size": 10,
    })

    # ระบุตำแหน่งไฟล์รูปภาพ
    image_path = Path(__file__).parent / "images" / "sample.jpg"

    # อ่านภาพต้นฉบับ
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise FileNotFoundError(
            f"ไม่พบรูปภาพ กรุณาตรวจสอบไฟล์: {image_path}"
        )

    # แปลงรูปแบบสีจาก BGR เป็น RGB
    image_rgb = cv2.cvtColor(
        image_bgr, cv2.COLOR_BGR2RGB
    )

    # แปลงภาพสีเป็นภาพระดับเทา
    image_gray = cv2.cvtColor(
        image_bgr, cv2.COLOR_BGR2GRAY
    )

    # ใช้ Gaussian Blur ลดรายละเอียดและสัญญาณรบกวน
    gaussian = cv2.GaussianBlur(
        image_rgb, (5, 5), 0
    )

    # ใช้ Median Blur ลดสัญญาณรบกวนแบบจุด
    median = cv2.medianBlur(image_rgb, 5)

    print("=" * 55)
    print("IMAGE FILTERING AND HISTOGRAM ANALYSIS")
    print("=" * 55)
    print(f"Image: {image_path.name}")
    print(f"Image dimensions: {image_rgb.shape[1]} x {image_rgb.shape[0]}")
    print("Filters applied: Gaussian Blur, Median Blur")

    # ส่วนที่ 1: เปรียบเทียบผลการกรองภาพ
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5),
        facecolor="#FFF8FC"
    )

    fig.suptitle(
        "IMAGE FILTERING LAB",
        fontsize=19,
        fontweight="bold",
        color="#D957A6",
        y=0.98
    )

    images = [image_rgb, gaussian, median]
    titles = [
        "01  Original Image",
        "02  Gaussian Blur",
        "03  Median Blur"
    ]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(
            title,
            fontsize=11,
            fontweight="bold",
            color="#247BA0",
            pad=12
        )
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.show()

    # ส่วนที่ 2: วิเคราะห์การกระจายความเข้มของพิกเซล
    fig, ax = plt.subplots(
        figsize=(12, 5),
        facecolor="#FFF8FC"
    )

    # คำนวณ Histogram ของภาพระดับเทา
    histogram = cv2.calcHist(
        [image_gray], [0], None, [256], [0, 256]
    )

    # วาดกราฟด้วยสีฟ้า Cyan
    ax.plot(
        np.arange(256),
        histogram.flatten(),
        color="#35BDE8",
        linewidth=2
    )

    # เติมพื้นที่ใต้กราฟด้วยสีฟ้าอ่อน
    ax.fill_between(
        np.arange(256),
        histogram.flatten(),
        color="#70D6FF",
        alpha=0.28
    )

    ax.set_title(
        "GRAYSCALE INTENSITY HISTOGRAM",
        fontsize=15,
        fontweight="bold",
        color="#D957A6",
        pad=15
    )
    ax.set_xlabel("Pixel Intensity (0 = Black, 255 = White)")
    ax.set_ylabel("Number of Pixels")
    ax.set_xlim(0, 255)

    # เพิ่มเส้นตารางให้อ่านค่าได้ง่าย
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )

    # ปรับกรอบกราฟให้ดูเรียบร้อย
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

    print("\nImage processing completed successfully.")


if __name__ == "__main__":
    main()