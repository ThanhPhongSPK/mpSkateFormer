import os
import csv
import numpy as np
import torch
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)

class Metrics:
    def __init__(self, result_folder, num_classes=60, class_names=None):
        """
        Args:
            result_folder: Path to save metrics files
            num_classes: Number of classes (default 60)
            class_names: Optional dict/list of class names for readable output
        """
        self.result_folder = result_folder
        self.num_classes = num_classes
        self.class_names = class_names
        
        # CSV file paths
        self.metrics_csv = os.path.join(result_folder, "metrics.csv")
        
        # Initialize metrics CSV with header
        self._init_metrics_csv()
        
        # Storage for collecting predictions during epoch
        self.reset()
    
    def _init_metrics_csv(self):
        """Initialize the metrics CSV file with headers"""
        if not os.path.exists(self.metrics_csv):
            with open(self.metrics_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch',
                    'train_loss', 'val_loss',
                    'train_acc', 'val_acc',
                    'precision_macro', 'recall_macro', 'f1_macro',
                    'top5_acc'
                ])
    
    def reset(self):
        """Reset collectors for new epoch"""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def collect(self, outputs, targets):
        """
        Collect predictions and labels during validation/test
        
        Args:
            outputs: Model outputs (logits), shape (batch_size, num_classes)
            targets: Ground truth labels, shape (batch_size,)
        """
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(targets.cpu().numpy())
        self.all_probs.extend(probs.cpu().numpy())
    
    def _top_k_accuracy(self, y_true, y_probs, k=5):
        """Calculate top-k accuracy"""
        top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
        correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        return 100.0 * np.mean(correct)
    
    def save_epoch_metrics(self, epoch, train_loss, val_loss, train_acc):
        """Save metrics for one epoch to CSV"""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        y_probs = np.array(self.all_probs)
        
        # Compute metrics
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        top1_acc = 100.0 * np.mean(y_true == y_pred)
        top5_acc = self._top_k_accuracy(y_true, y_probs, k=5)
        
        with open(self.metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{train_acc:.2f}",
                f"{top1_acc:.2f}",
                f"{precision_macro:.4f}",
                f"{recall_macro:.4f}",
                f"{f1_macro:.4f}",
                f"{top5_acc:.2f}"
            ])
        
        # Reset for next epoch
        self.reset()
        
        return top1_acc

    def save_confusion_matrix(self):
        """Save final confusion matrix at end of training"""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        cm_path = os.path.join(self.result_folder, "confusion_matrix.csv")
        
        # Create header
        if self.class_names:
            header = [''] + [self.class_names.get(i, f"Class_{i}") for i in range(self.num_classes)]
        else:
            header = [''] + [f"Class_{i}" for i in range(self.num_classes)]
        
        with open(cm_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for i, row in enumerate(cm):
                row_label = self.class_names.get(i, f"Class_{i}") if self.class_names else f"Class_{i}"
                writer.writerow([row_label] + row.tolist())
        
        print(f"Confusion matrix saved to {cm_path}")

def setup_logger(result_folder):
    """
    Creates a logger that writes to both console and a log file.
    """
    log_file_path = os.path.join(result_folder, "train.log")
    
    def log(msg):
        print(msg)  # Print to console
        with open(log_file_path, "a") as f:
            f.write(str(msg) + "\n")  # Write to file
            
    return log