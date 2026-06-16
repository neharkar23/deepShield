import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no Tk/display required)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from tqdm import tqdm

# Add project root directory to sys.path to allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import logger, set_seed
from src.data_loader import get_data_loaders
from src.train import get_model

class Evaluator:
    """Evaluates trained models and generates visualization plots and reports."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        set_seed(Config.SEED)
        
        # Load best model checkpoint
        self.model = get_model(self.model_name, pretrained=False)
        self.best_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"{self.model_name}_best.pth")
        
        if not os.path.exists(self.best_ckpt_path):
            raise FileNotFoundError(f"Trained checkpoint not found at {self.best_ckpt_path}. Train the model first.")
            
        logger.info(f"Loading best weights from {self.best_ckpt_path}")
        checkpoint = torch.load(self.best_ckpt_path, map_location=Config.DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(Config.DEVICE)
        self.model.eval()

    def run_evaluation(self):
        """Runs model inference on the test set, computes metrics, and saves reports."""
        _, _, test_loader = get_data_loaders()
        
        if test_loader is None or len(test_loader.dataset) == 0:
            logger.error("Test dataset folder is empty or not found.")
            return
            
        logger.info(f"Evaluating {self.model_name} on test set...")
        
        all_targets = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(Config.DEVICE)
                logits = self.model(images)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= 0.5).astype(float)
                
                all_targets.extend(labels.numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)
                
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_targets, all_preds, all_probs)
        
        # Save plots
        self.plot_confusion_matrix(all_targets, all_preds)
        self.plot_roc_curve(all_targets, all_probs, metrics["roc_auc"])
        self.plot_precision_recall_curve(all_targets, all_probs, metrics["ap"])
        self.plot_training_curves()
        
        # Save metrics report
        self.save_text_report(all_targets, all_preds, metrics)
        
        logger.info(f"Evaluation complete. Results saved in outputs/")
        return metrics

    def calculate_metrics(self, y_true, y_pred, y_prob) -> dict:
        """Calculates classification metrics."""
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "ap": ap
        }

    def plot_confusion_matrix(self, y_true, y_pred):
        """Generates and saves the confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"]
        )
        plt.title(f"Confusion Matrix - {self.model_name.upper()}")
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        
        out_path = os.path.join(Config.CONF_MATRIX_DIR, f"{self.model_name}_cm.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix to {out_path}")

    def plot_roc_curve(self, y_true, y_prob, auc_score):
        """Generates and saves the Receiver Operating Characteristic (ROC) curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {self.model_name.upper()}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        out_path = os.path.join(Config.ROC_DIR, f"{self.model_name}_roc.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved ROC curve to {out_path}")

    def plot_precision_recall_curve(self, y_true, y_prob, ap_score):
        """Generates and saves the Precision-Recall (PR) curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color="forestgreen", lw=2, label=f"PR curve (AP = {ap_score:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {self.model_name.upper()}")
        plt.legend(loc="lower left")
        plt.tight_layout()
        
        out_path = os.path.join(Config.PLOTS_DIR, f"{self.model_name}_pr.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved Precision-Recall curve to {out_path}")

    def plot_training_curves(self):
        """Loads model training logs (JSON) and plots accuracy and loss curves."""
        history_path = os.path.join(Config.REPORTS_DIR, f"{self.model_name}_history.json")
        if not os.path.exists(history_path):
            logger.warning(f"Training history file not found at {history_path}. Skipping training curves plotting.")
            return
            
        with open(history_path, "r") as f:
            history = json.load(f)
            
        epochs = list(range(len(history["train_loss"])))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss Curves
        ax1.plot(epochs, history["train_loss"], label="Train Loss", color="royalblue", marker='o')
        ax1.plot(epochs, history["val_loss"], label="Val Loss", color="orange", marker='x')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy Curves
        ax2.plot(epochs, history["train_acc"], label="Train Acc", color="royalblue", marker='o')
        ax2.plot(epochs, history["val_acc"], label="Val Acc", color="orange", marker='x')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        out_path = os.path.join(Config.PLOTS_DIR, f"{self.model_name}_curves.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved training curves to {out_path}")

    def save_text_report(self, y_true, y_pred, metrics: dict):
        """Saves a detailed performance report in a text file."""
        report_str = classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4)
        out_path = os.path.join(Config.REPORTS_DIR, f"{self.model_name}_report.txt")
        
        with open(out_path, "w") as f:
            f.write(f"DEEPSHIELD EVALUATION REPORT - {self.model_name.upper()}\n")
            f.write("="*60 + "\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.6f}\n")
            f.write(f"Precision: {metrics['precision']:.6f}\n")
            f.write(f"Recall:    {metrics['recall']:.6f}\n")
            f.write(f"F1 Score:  {metrics['f1']:.6f}\n")
            f.write(f"ROC-AUC:   {metrics['roc_auc']:.6f}\n")
            f.write(f"Avg Precision (AP): {metrics['ap']:.6f}\n")
            f.write("="*60 + "\n\n")
            f.write("CLASSIFICATION REPORT:\n")
            f.write(report_str)
            
        logger.info(f"Saved evaluation metrics report to {out_path}")
        print("\n" + "="*50)
        print(f"      EVALUATION METRICS: {self.model_name.upper()}")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"Precision: {metrics['precision']*100:.2f}%")
        print(f"Recall:    {metrics['recall']*100:.2f}%")
        print(f"F1 Score:  {metrics['f1']*100:.2f}%")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection models.")
    parser.add_argument("--model", type=str, default="hybrid", choices=["efficientnet_b0", "efficientnet_b3", "xception", "vit", "hybrid"], help="Model to evaluate")
    args = parser.parse_args()
    
    evaluator = Evaluator(model_name=args.model)
    evaluator.run_evaluation()
