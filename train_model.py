import os
import sys
import json
import csv
import argparse
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import transforms


class TeeLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            return False, True

        self.counter += 1
        should_stop = self.counter >= self.patience
        return should_stop, False


class CalligraphyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.char_to_idx = {}
        self.idx_to_char = {}

        supported_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
        char_idx = 0
        for char_name in sorted(os.listdir(data_dir)):
            char_dir = os.path.join(data_dir, char_name)
            if os.path.isdir(char_dir):
                if char_name not in self.char_to_idx:
                    self.char_to_idx[char_name] = char_idx
                    self.idx_to_char[char_idx] = char_name
                    char_idx += 1

                for root, _, files in os.walk(char_dir):
                    for file in files:
                        if file.lower().endswith(supported_exts):
                            self.image_paths.append(os.path.join(root, file))
                            self.labels.append(self.char_to_idx[char_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Skipping corrupted image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_model(num_classes):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        loss = checkpoint['loss']

        print(f"Resuming training from epoch {start_epoch + 1}, best accuracy: {best_acc:.4f}")
        return start_epoch, best_acc, loss

    print(f"No checkpoint found at {checkpoint_path}, starting training from scratch.")
    return 0, 0.0, float('inf')


def train_one_epoch(model, dataloader, criterion, optimizer, device, start_batch_idx=0):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, (inputs, labels) in enumerate(dataloader):
        if i < start_batch_idx:
            continue

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

        if i % 50 == 49:
            print(f"  Batch {i + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = correct_predictions.double() / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, measure_inference=False):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    y_true = []
    y_pred = []
    inference_time_seconds = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_start = time.perf_counter()
            outputs = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            if measure_inference:
                inference_time_seconds += time.perf_counter() - batch_start
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = correct_predictions.double() / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc, y_true, y_pred, inference_time_seconds


def calculate_classification_metrics(y_true, y_pred, num_classes):
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    pair_counts = Counter(zip(y_true, y_pred))

    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    macro_f1_total = 0.0
    total_samples = len(y_true)
    correct_samples = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)

    for class_idx in range(num_classes):
        tp = pair_counts.get((class_idx, class_idx), 0)
        predicted = pred_counts.get(class_idx, 0)
        actual = true_counts.get(class_idx, 0)

        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        weighted_precision += precision * actual
        weighted_recall += recall * actual
        weighted_f1 += f1 * actual
        macro_f1_total += f1

    metrics = {
        'accuracy': correct_samples / total_samples if total_samples else 0.0,
        'precision': weighted_precision / total_samples if total_samples else 0.0,
        'recall': weighted_recall / total_samples if total_samples else 0.0,
        'f1': weighted_f1 / total_samples if total_samples else 0.0,
        'macro_f1': macro_f1_total / num_classes if num_classes else 0.0,
        'num_samples': total_samples,
    }
    return metrics


def save_metrics(history, csv_path, json_path):
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'epoch',
                'train_loss',
                'train_acc',
                'val_loss',
                'val_acc',
                'best_val_acc_so_far',
                'epoch_time_seconds',
                'learning_rate',
            ]
        )
        writer.writeheader()
        writer.writerows(history)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_best_model_metrics(metrics, output_dir):
    metrics_json_path = os.path.join(output_dir, 'best_model_metrics.json')
    metrics_txt_path = os.path.join(output_dir, 'best_model_metrics.txt')

    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(metrics_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Epoch: {metrics['epoch']}\n")
        f.write(f"Samples: {metrics['num_samples']}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-score: {metrics['f1']:.4f}\n")
        f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")
        f.write(f"Val Loss: {metrics['val_loss']:.4f}\n")
        f.write(f"Val Acc: {metrics['val_acc']:.4f}\n")
        f.write(f"Total Parameters: {metrics['total_parameters']}\n")
        f.write(f"Trainable Parameters: {metrics['trainable_parameters']}\n")
        f.write(f"Inference Time Total (s): {metrics['inference_time_seconds']:.6f}\n")
        f.write(f"Inference Time Per Sample (ms): {metrics['inference_time_per_sample_ms']:.6f}\n")

    return metrics_json_path, metrics_txt_path


def save_training_summary(summary, output_dir):
    summary_json_path = os.path.join(output_dir, 'training_summary.json')
    summary_txt_path = os.path.join(output_dir, 'training_summary.txt')

    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Total Parameters: {summary['total_parameters']}\n")
        f.write(f"Trainable Parameters: {summary['trainable_parameters']}\n")
        f.write(f"Total Training Time (s): {summary['total_training_time_seconds']:.6f}\n")
        f.write(f"Average Epoch Time (s): {summary['average_epoch_time_seconds']:.6f}\n")
        f.write(f"Completed Epochs: {summary['completed_epochs']}\n")

    return summary_json_path, summary_txt_path


def main():
    parser = argparse.ArgumentParser(description='Train a calligraphy recognition model.')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the chinese_fonts directory.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint-freq', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--log-dir', type=str, default='output/logs', help='Directory to save training logs and metrics.')
    parser.add_argument('--model-dir', type=str, default='output/models', help='Directory to save model weights and checkpoints.')
    parser.add_argument('--lr-patience', type=int, default=3, help='ReduceLROnPlateau patience based on validation loss.')
    parser.add_argument('--lr-factor', type=float, default=0.5, help='ReduceLROnPlateau decay factor.')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum learning rate for ReduceLROnPlateau.')
    parser.add_argument('--early-stop-patience', type=int, default=8, help='Early stopping patience based on validation loss.')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0, help='Minimum validation loss improvement for early stopping.')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, 'train.log')
    metrics_csv_path = os.path.join(args.log_dir, 'train_metrics.csv')
    metrics_json_path = os.path.join(args.log_dir, 'train_metrics.json')
    best_model_path = os.path.join(args.model_dir, 'best_model.pth')
    latest_checkpoint_path = os.path.join(args.model_dir, 'latest_checkpoint.pth')

    original_stdout = sys.stdout
    tee_logger = TeeLogger(log_path)
    sys.stdout = tee_logger
    history = []
    total_training_start = time.perf_counter()

    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if not torch.cuda.is_available():
            print('Warning: CUDA not available. Training on CPU will be very slow.')

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.7, 1.3)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        full_dataset = CalligraphyDataset(args.data_dir, transform=data_transforms['train'])
        num_classes = len(full_dataset.char_to_idx)
        print(f"Found {len(full_dataset)} images belonging to {num_classes} classes.")

        if not os.path.exists('char_map.json'):
            with open('char_map.json', 'w', encoding='utf-8') as f:
                json.dump(full_dataset.char_to_idx, f, ensure_ascii=False, indent=4)
            print('Character map saved to char_map.json')
        else:
            print('Character map already exists, skipping creation.')

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        val_dataset.dataset = CalligraphyDataset(args.data_dir, transform=data_transforms['val'])

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4),
            'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        }

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Training log will be saved to: {log_path}")
        print(f"Model artifacts will be saved to: {args.model_dir}")

        model = get_model(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
        early_stopper = EarlyStopping(
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
        )
        total_parameters = sum(p.numel() for p in model.parameters())
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_parameters}")
        print(f"Trainable parameters: {trainable_parameters}")
        print(
            f"ReduceLROnPlateau enabled: factor={args.lr_factor}, "
            f"patience={args.lr_patience}, min_lr={args.min_lr}"
        )
        print(
            f"EarlyStopping enabled: patience={args.early_stop_patience}, "
            f"min_delta={args.early_stop_min_delta}"
        )

        start_epoch = 0
        best_acc = 0.0
        if args.resume:
            start_epoch, best_acc, _ = load_checkpoint(args.resume, model, optimizer, scheduler)

        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.perf_counter()
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print('-' * 10)

            train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
            print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

            val_loss, val_acc, y_true, y_pred, inference_time_seconds = evaluate(
                model, dataloaders['val'], criterion, device, measure_inference=True
            )
            print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.8f}")
            if len(y_true) > 0:
                print(
                    "Validation inference time: "
                    f"{inference_time_seconds:.4f}s total, "
                    f"{(inference_time_seconds / len(y_true)) * 1000:.4f} ms/sample"
                )

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"Learning rate reduced to: {new_lr:.8f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f'Best model saved to {best_model_path}')
                best_model_metrics = calculate_classification_metrics(y_true, y_pred, num_classes)
                best_model_metrics.update({
                    'epoch': epoch + 1,
                    'val_loss': float(val_loss),
                    'val_acc': float(val_acc),
                    'best_model_path': best_model_path,
                    'total_parameters': int(total_parameters),
                    'trainable_parameters': int(trainable_parameters),
                    'inference_time_seconds': float(inference_time_seconds),
                    'inference_time_per_sample_ms': float((inference_time_seconds / len(y_true)) * 1000 if y_true else 0.0),
                })
                best_metrics_json_path, best_metrics_txt_path = save_best_model_metrics(best_model_metrics, args.log_dir)
                print(
                    "Best model metrics saved to "
                    f"{best_metrics_json_path} and {best_metrics_txt_path}"
                )

            epoch_time_seconds = time.perf_counter() - epoch_start
            print(f"Epoch time: {epoch_time_seconds:.2f}s")

            history.append({
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'train_acc': float(train_acc),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
                'best_val_acc_so_far': float(best_acc),
                'epoch_time_seconds': float(epoch_time_seconds),
                'learning_rate': float(optimizer.param_groups[0]['lr']),
            })
            save_metrics(history, metrics_csv_path, metrics_json_path)

            if (epoch + 1) % args.checkpoint_freq == 0:
                checkpoint_path = os.path.join(args.model_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                    'loss': val_loss,
                }, checkpoint_path)

            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'loss': val_loss,
            }, latest_checkpoint_path)

            should_stop, improved = early_stopper.step(float(val_loss))
            if improved:
                print("EarlyStopping monitor: validation loss improved.")
            else:
                print(
                    "EarlyStopping monitor: no validation loss improvement "
                    f"({early_stopper.counter}/{early_stopper.patience})."
                )
            if should_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        total_training_time_seconds = time.perf_counter() - total_training_start
        average_epoch_time_seconds = (
            sum(item['epoch_time_seconds'] for item in history) / len(history) if history else 0.0
        )
        training_summary = {
            'total_parameters': int(total_parameters),
            'trainable_parameters': int(trainable_parameters),
            'total_training_time_seconds': float(total_training_time_seconds),
            'average_epoch_time_seconds': float(average_epoch_time_seconds),
            'completed_epochs': len(history),
        }
        summary_json_path, summary_txt_path = save_training_summary(training_summary, args.log_dir)

        print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")
        print(f"Total training time: {total_training_time_seconds:.2f}s")
        print(f"Average epoch time: {average_epoch_time_seconds:.2f}s")
        print(f"Saved epoch metrics CSV to {metrics_csv_path}")
        print(f"Saved epoch metrics JSON to {metrics_json_path}")
        print(f"Latest checkpoint saved to {latest_checkpoint_path}")
        print(f"Saved training summary to {summary_json_path} and {summary_txt_path}")
    finally:
        sys.stdout = original_stdout
        tee_logger.close()


if __name__ == '__main__':
    main()
