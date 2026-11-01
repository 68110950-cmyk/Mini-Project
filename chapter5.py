"""บททดลอง: เปรียบเทียบ learning rate และ dropout บน MNIST

ติดตั้งไลบรารี:
    python3 -m pip install torch torchvision matplotlib numpy

รัน:
    python3 mnist_hyperparameter_lab.py

ข้อมูลจะดาวน์โหลดลง ./data และผลลัพธ์ลง ./artifacts
สคริปต์นี้ไม่ต้องติดตั้ง pandas
"""

import copy
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# =============================================================================
# 1. ค่าตั้งต้น — เปลี่ยนค่าการทดลองได้จากส่วนนี้
# =============================================================================
SEED = 42
BATCH_SIZE = 128
EPOCHS = 4
LEARNING_RATES = [1e-3, 1e-4]
DROPOUT_RATES = [0.0, 0.3]
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./artifacts")


# =============================================================================
# 2. ทำให้การทดลองทำซ้ำได้ และเลือกอุปกรณ์
# =============================================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# 3. โมเดล CNN สำหรับจำแนกตัวเลข 0–9
# =============================================================================
class MNISTCNN(nn.Module):
    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))  # [B, 1, 28, 28] -> [B, 16, 14, 14]
        x = self.pool(self.relu(self.conv2(x)))  # -> [B, 32, 7, 7]
        x = torch.flatten(x, start_dim=1)  # -> [B, 1568]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # ใช้เฉพาะระหว่าง train; ปิดอัตโนมัติใน eval
        return self.fc2(x)  # logits รูปร่าง [B, 10]


# =============================================================================
# 4. ฟังก์ชันฝึกหนึ่ง epoch และวัดผล
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        count = labels.size(0)
        total_loss += loss.item() * count
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += count

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

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


# =============================================================================
# 5. ทดลองหนึ่ง configuration (คู่ learning rate + dropout)
# =============================================================================
def run_experiment(config, run_id, train_dataset, val_dataset, device):
    # เริ่มทุก configuration จาก seed เดียวกันเพื่อให้เปรียบเทียบได้ยุติธรรมขึ้น
    set_seed(SEED)
    train_generator = torch.Generator().manual_seed(SEED)

    # num_workers=0 ทำงานเสถียรบน macOS, Windows และ Jupyter
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = MNISTCNN(dropout_p=config["dropout_p"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    history = {
        "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
    }
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        # เลือก checkpoint ที่ validation accuracy สูงสุด; ถ้าเสมอเลือก loss ต่ำกว่า
        if val_acc > best_val_accuracy or (
            val_acc == best_val_accuracy and val_loss < best_val_loss
        ):
            best_val_accuracy = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Run {run_id} | Epoch {epoch + 1}/{EPOCHS} | "
            f"lr={config['lr']:.0e}, dropout={config['dropout_p']:.1f} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.2%} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.2%}"
        )

    return {
        "config": config,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "best_state": best_state,
    }


# =============================================================================
# 6. สรุปผลและวาดกราฟเปรียบเทียบ
# =============================================================================
def save_summary_csv(results, path: Path) -> None:
    # ใช้ csv ของ Python แทน pandas เพื่อลดไลบรารีที่ต้องติดตั้ง
    sorted_results = sorted(
        results,
        key=lambda result: (
            -result["best_val_accuracy"], result["best_val_loss"]
        ),
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "run", "learning_rate", "dropout_p", "best_epoch",
            "best_val_accuracy", "val_loss_at_best",
        ])
        for run_id, result in enumerate(sorted_results, start=1):
            writer.writerow([
                run_id,
                result["config"]["lr"],
                result["config"]["dropout_p"],
                result["best_epoch"],
                result["best_val_accuracy"],
                result["best_val_loss"],
            ])


def plot_results(results) -> None:
    # กราฟ loss และ accuracy: หนึ่งช่องต่อหนึ่ง configuration
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, result in enumerate(results):
        row, col = divmod(i, 2)
        history = result["history"]
        epochs = range(1, EPOCHS + 1)
        config = result["config"]
        title = f"lr={config['lr']:.0e}, dropout={config['dropout_p']:.1f}"

        axes[row, col].plot(epochs, history["train_loss"], marker="o", label="Train")
        axes[row, col].plot(epochs, history["val_loss"], marker="o", label="Validation")
        axes[row, col].set_title(f"Loss | {title}")
        axes[row, col].set_xlabel("Epoch")
        axes[row, col].set_ylabel("Cross-entropy loss")
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_comparison.png", dpi=150)
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, result in enumerate(results):
        row, col = divmod(i, 2)
        history = result["history"]
        epochs = range(1, EPOCHS + 1)
        config = result["config"]
        title = f"lr={config['lr']:.0e}, dropout={config['dropout_p']:.1f}"

        axes[row, col].plot(epochs, history["train_accuracy"], marker="o", label="Train")
        axes[row, col].plot(epochs, history["val_accuracy"], marker="o", label="Validation")
        axes[row, col].set_title(f"Accuracy | {title}")
        axes[row, col].set_xlabel("Epoch")
        axes[row, col].set_ylabel("Accuracy")
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=150)
    plt.show()


# =============================================================================
# 7. เริ่มโปรแกรม: โหลดข้อมูล -> ทดลอง -> เลือกโมเดลที่ดีที่สุด -> บันทึก
# =============================================================================
def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch:", torch.__version__)
    print("อุปกรณ์:", device)

    # โหลดเฉพาะ train split ของ MNIST แล้วแยกเป็น train 55,000 / validation 5,000
    transform = transforms.ToTensor()
    full_train_dataset = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [55_000, 5_000],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"Train: {len(train_dataset):,}")
    print(f"Validation: {len(val_dataset):,}")

    # สร้าง 4 การทดลองจาก learning rate 2 ค่า x dropout 2 ค่า
    configs = [
        {"lr": lr, "dropout_p": dropout_p}
        for lr in LEARNING_RATES
        for dropout_p in DROPOUT_RATES
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for run_id, config in enumerate(configs, start=1):
        print(f"\nเริ่ม configuration {run_id}/{len(configs)}: {config}")
        results.append(
            run_experiment(config, run_id, train_dataset, val_dataset, device)
        )

    # เรียงผล: validation accuracy มากก่อน และ loss น้อยใช้ตัดสินเมื่อคะแนนเสมอ
    results.sort(key=lambda result: (
        -result["best_val_accuracy"], result["best_val_loss"]
    ))
    print("\nสรุปผลเรียงตาม validation accuracy:")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. lr={result['config']['lr']:.0e}, "
            f"dropout={result['config']['dropout_p']:.1f}, "
            f"epoch={result['best_epoch']}, "
            f"val accuracy={result['best_val_accuracy']:.2%}, "
            f"val loss={result['best_val_loss']:.4f}"
        )

    best_result = results[0]
    best_model = MNISTCNN(
        dropout_p=best_result["config"]["dropout_p"]
    ).to(device)
    best_model.load_state_dict(best_result["best_state"])
    best_model.eval()

    # บันทึกน้ำหนักโมเดลที่ดีที่สุดและตารางผลการทดลอง
    model_path = OUTPUT_DIR / "best_mnist_cnn.pth"
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "config": best_result["config"],
            "best_epoch": best_result["best_epoch"],
            "best_val_accuracy": best_result["best_val_accuracy"],
            "best_val_loss": best_result["best_val_loss"],
        },
        model_path,
    )
    summary_path = OUTPUT_DIR / "experiment_summary.csv"
    save_summary_csv(results, summary_path)
    plot_results(results)

    print(f"\nบันทึกโมเดลที่ดีที่สุด: {model_path}")
    print(f"บันทึกตารางสรุป: {summary_path}")
    print(f"บันทึกกราฟในโฟลเดอร์: {OUTPUT_DIR}")


if __name__ == "__main__":
    # จุดเริ่มต้นโปรแกรม; จำเป็นสำหรับการรันเป็นไฟล์บน Windows/macOS
    main()
