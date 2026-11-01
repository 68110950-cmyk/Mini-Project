"""บทเรียน: สร้างและฝึก CNN จำแนกภาพตัวเลข MNIST ด้วย PyTorch

ติดตั้งไลบรารีก่อน:
    python -m pip install torch torchvision matplotlib

รันจากโฟลเดอร์ที่เก็บไฟล์นี้:
    python mnist_cnn_tutorial.py

ครั้งแรกจะดาวน์โหลด MNIST ไว้ในโฟลเดอร์ ./data และบันทึกโมเดลไว้ที่
./artifacts/mnist_cnn.pth
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------------------------------------------------------
# 1. ตั้งค่าการทดลอง — แก้ค่าที่นี่เพื่อเปลี่ยนการฝึก
# -----------------------------------------------------------------------------
SEED = 42  # ค่าเริ่มต้นสุ่ม ช่วยให้เปรียบเทียบการทดลองได้
BATCH_SIZE = 128  # จำนวนภาพที่โมเดลดูต่อหนึ่งรอบปรับน้ำหนัก
EPOCHS = 3  # จำนวนรอบที่โมเดลเห็นข้อมูลฝึกทั้งหมด
LEARNING_RATE = 1e-3  # ขนาดก้าวในการปรับน้ำหนักโมเดล
DATA_DIR = Path("./data")  # ที่เก็บชุดข้อมูล MNIST ที่ดาวน์โหลด
ARTIFACT_DIR = Path("./artifacts")  # ที่เก็บไฟล์ผลลัพธ์ เช่น น้ำหนักโมเดล
MODEL_PATH = ARTIFACT_DIR / "mnist_cnn.pth"  # ชื่อและตำแหน่งไฟล์โมเดล


def set_seed(seed: int = SEED) -> None:
    """กำหนด seed ให้ผลการทดลองทำซ้ำได้ง่ายขึ้น (อาจไม่เหมือนกันทุกอุปกรณ์)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# 2. เตรียมข้อมูล — ดาวน์โหลด แปลงภาพ และจัดภาพเป็น batch
# -----------------------------------------------------------------------------
def make_data_loaders(device: torch.device):
    # แปลงภาพเป็น Tensor รูปร่าง [channel, height, width] และค่าพิกเซล 0 ถึง 1
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    # DataLoader จัดชุดภาพให้โมเดล; num_workers=0 ช่วยให้เริ่มง่ายบนทุกระบบ
    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    return train_dataset, test_dataset, train_loader, test_loader


def show_sample_batch(loader: DataLoader) -> None:
    """ดูภาพตัวอย่างและตรวจรูปร่างข้อมูลก่อนเริ่มฝึก"""
    images, labels = next(iter(loader))
    print("รูปร่างภาพใน batch:", tuple(images.shape), "# [B, 1, 28, 28]")
    print("รูปร่าง label:", tuple(labels.shape), "# [B]")
    print(f"ช่วงค่าพิกเซล: {images.min().item():.1f} ถึง {images.max().item():.1f}")

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0], cmap="gray")
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")
    fig.suptitle("ตัวอย่างภาพจากชุดฝึก")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 3. สร้างโมเดล — กำหนดชั้นที่แปลงภาพเป็นคะแนนของเลข 0–9
# -----------------------------------------------------------------------------
class MNISTCNN(nn.Module):
    """CNN ขนาดเล็ก: convolution -> pooling สองรอบ -> fully connected"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # ภาพเทา 1 ช่อง -> 16 แผนที่คุณลักษณะ
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16 -> 32 แผนที่คุณลักษณะ
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # ลดความกว้าง/สูงลงครึ่งหนึ่ง
        self.relu = nn.ReLU()  # เพิ่มความสามารถเรียนรู้ความสัมพันธ์ที่ไม่เป็นเส้นตรง
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # รับ feature หลัง pool สองรอบ
        self.fc2 = nn.Linear(128, 10)  # ส่งคะแนนออกมา 10 ค่า แทนเลข 0 ถึง 9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 1, 28, 28] -> [B, 16, 14, 14]
        x = self.pool(self.relu(self.conv1(x)))
        # [B, 16, 14, 14] -> [B, 32, 7, 7]
        x = self.pool(self.relu(self.conv2(x)))
        # รวม channel และตำแหน่งภาพเป็นเวกเตอร์ต่อหนึ่งภาพ
        x = torch.flatten(x, start_dim=1)  # [B, 1568]
        x = self.relu(self.fc1(x))
        return self.fc2(x)  # [B, 10] เป็น logits (ยังไม่ผ่าน softmax)

    @torch.no_grad()
    def get_first_layer_features(self, x: torch.Tensor) -> torch.Tensor:
        """คืน feature maps หลัง convolution ชั้นแรก สำหรับการแสดงผล"""
        return self.relu(self.conv1(x))


def inspect_shapes(model: MNISTCNN, device: torch.device) -> None:
    """ส่งภาพจำลองผ่านแต่ละชั้นเพื่อให้เห็นการเปลี่ยนขนาด tensor"""
    model.eval()
    x = torch.zeros(4, 1, 28, 28, device=device)
    print("\nตรวจสอบขนาด tensor:")
    print("Input:", tuple(x.shape))
    x = model.pool(model.relu(model.conv1(x)))
    print("หลัง conv1 + ReLU + pool:", tuple(x.shape))
    x = model.pool(model.relu(model.conv2(x)))
    print("หลัง conv2 + ReLU + pool:", tuple(x.shape))
    x = torch.flatten(x, start_dim=1)
    print("หลัง flatten:", tuple(x.shape))
    print("ผลลัพธ์ logits:", tuple(model(x.new_zeros(4, 1, 28, 28)).shape))


# -----------------------------------------------------------------------------
# 4. ฝึกและประเมินผล — คำนวณ loss, ปรับน้ำหนัก และวัดความแม่นยำ
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # เปิดโหมดฝึก
    total_loss = total_correct = total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()  # คำนวณ gradient
        optimizer.step()  # ปรับน้ำหนักโมเดล

        count = labels.size(0)
        total_loss += loss.item() * count
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += count

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()  # ปิดพฤติกรรมเฉพาะการฝึก และไม่คำนวณ gradient
    total_loss = total_correct = total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        count = labels.size(0)
        total_loss += loss.item() * count
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += count

    return total_loss / total_examples, total_correct / total_examples


def plot_history(history: dict) -> None:
    """วาดกราฟ loss และ accuracy ของชุดฝึก/ชุดทดสอบในแต่ละ epoch"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["test_loss"], marker="o", label="Test")
    axes[0].set(title="Loss by epoch", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train")
    axes[1].plot(epochs, history["test_accuracy"], marker="o", label="Test")
    axes[1].set(title="Accuracy by epoch", xlabel="Epoch", ylabel="Accuracy")
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 5. แสดงผลการเรียนรู้ — ดู feature maps และภาพที่โมเดลทายผิด
# -----------------------------------------------------------------------------
@torch.no_grad()
def show_feature_maps(model, dataset, device, index=0, channels=8):
    model.eval()
    image, label = dataset[index]
    features = model.get_first_layer_features(image.unsqueeze(0).to(device))
    features = features.squeeze(0).cpu()

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        if i < min(channels, features.shape[0]):
            ax.imshow(features[i], cmap="viridis")
            ax.set_title(f"Filter {i + 1}")
        ax.axis("off")
    fig.suptitle(f"ภาพจริง: {label} | Feature maps จาก conv1")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def show_misclassified(model, loader, device, max_items=10):
    model.eval()
    mistakes = []
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        for image, label, prediction in zip(images, labels, predictions):
            if prediction.item() != label.item():
                mistakes.append((image[0], label.item(), prediction.item()))
                if len(mistakes) == max_items:
                    break
        if len(mistakes) == max_items:
            break

    print(f"พบภาพที่ทำนายผิดและนำมาแสดง: {len(mistakes)} ภาพ")
    if not mistakes:
        return
    cols = 5
    rows = (len(mistakes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows), squeeze=False)
    for ax, (image, true_label, prediction) in zip(axes.flat, mistakes):
        ax.imshow(image, cmap="gray")
        ax.set_title(f"จริง: {true_label} | ทาย: {prediction}")
        ax.axis("off")
    for ax in axes.flat[len(mistakes):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """ลำดับหลักของโปรแกรม: เตรียมข้อมูล -> สร้าง -> ฝึก -> ประเมิน -> บันทึก"""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch:", torch.__version__)
    print("อุปกรณ์ที่ใช้:", device)

    train_dataset, test_dataset, train_loader, test_loader = make_data_loaders(device)
    print(f"จำนวนภาพ: train={len(train_dataset):,}, test={len(test_dataset):,}")
    show_sample_batch(train_loader)

    model = MNISTCNN().to(device)
    print("\nโครงสร้างโมเดล:\n", model)
    inspect_shapes(model, device)

    criterion = nn.CrossEntropyLoss()  # เปรียบเทียบ logits กับ label จริง
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # วิธีปรับน้ำหนัก
    history = {
        "train_loss": [], "train_accuracy": [],
        "test_loss": [], "test_accuracy": [],
    }

    # วนฝึกครบตามจำนวน epoch แล้ววัดผลบนชุดทดสอบทุกครั้ง
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_acc)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train loss={train_loss:.4f}, accuracy={train_acc:.2%} | "
            f"test loss={test_loss:.4f}, accuracy={test_acc:.2%}"
        )

    # บันทึกเฉพาะน้ำหนัก เพื่อโหลดกลับเข้า MNISTCNN ได้ภายหลัง
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nบันทึกน้ำหนักโมเดลแล้ว: {MODEL_PATH}")

    # สรุปผลเป็นกราฟและภาพตัวอย่าง
    plot_history(history)
    show_feature_maps(model, test_dataset, device, index=0)
    show_misclassified(model, test_loader, device)


if __name__ == "__main__":
    # จำเป็นสำหรับการรันเป็นสคริปต์ โดยเฉพาะบน Windows
    main()
