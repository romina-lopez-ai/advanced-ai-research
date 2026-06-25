"""
COMP6011 Research Task 3 — ECG Experiment Template
Student: <!-- Your name -->
Date: <!-- Date -->
Description: <!-- What this experiment tests -->

Experiment Log:
- Run 1: Date | Model | Dataset | Accuracy | Notes
- Run 2: Date | Model | Dataset | Accuracy | Notes
"""

# =============================================================================
# Configuration — update and commit each time you run an experiment
# =============================================================================
CONFIG = {
    "experiment_name": "baseline_resnet1d",
    "model": "ResNet1D",
    "dataset": "PTB-XL",
    "sampling_rate": 500,       # Hz
    "signal_length": 5000,      # samples (10 seconds at 500Hz)
    "num_leads": 12,
    "num_classes": 7,
    "classes": ["NORM", "AFIB", "AFLT", "1dAVb", "RBBB", "LBBB", "OTHERS"],
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 1e-3,
    "confidence_threshold": 0.70,  # Below this → flag for manual review
    "notes": "Describe what you are testing in this run"
}

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

print(f"Experiment: {CONFIG['experiment_name']}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"PyTorch: {torch.__version__}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data Loading
# =============================================================================

def load_ecg_data(dataset_path, split="train"):
    """
    Load and preprocess ECG data.
    
    Returns:
        signals: (N, num_leads, signal_length) numpy array
        labels: (N,) integer labels
    """
    # TODO: implement your data loading
    # Recommended datasets:
    # - PTB-XL: physionet.org/content/ptb-xl
    # - PhysioNet 2020: physionet.org/content/challenge-2020
    raise NotImplementedError("Implement data loading for your chosen dataset")


def preprocess_signal(signal, fs=500):
    """
    Preprocess a raw ECG signal.
    
    Steps to consider:
    1. Bandpass filter (0.5 - 40 Hz) to remove noise
    2. Normalise per lead (zero mean, unit variance)
    3. Handle NaN/missing values
    """
    # TODO: implement preprocessing
    raise NotImplementedError("Implement your preprocessing pipeline")


# =============================================================================
# Model Definition
# =============================================================================

class ECGClassifier(nn.Module):
    """
    Your ECG classification model.
    Replace this with your chosen architecture.
    Document why you made each architectural choice.
    """
    def __init__(self, num_leads=12, num_classes=7):
        super().__init__()
        # TODO: define your architecture
        # Common choices:
        # - 1D CNN / ResNet for raw signal
        # - Transformer for sequence modelling
        # - Hybrid CNN-LSTM
        raise NotImplementedError("Implement your model architecture")
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_leads, signal_length)
        Returns:
            logits: (batch, num_classes)
        """
        raise NotImplementedError("Implement forward pass")


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for signals, labels in dataloader:
        signals, labels = signals.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    
    return total_loss / len(dataloader), correct / total


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, dataloader):
    """
    Evaluate model and return per-class metrics.
    Reports accuracy, sensitivity, specificity per class.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(DEVICE)
            logits = model(signals)
            probs = torch.softmax(logits, dim=1)
            
            confidence, pred = probs.max(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidence.cpu().numpy())
    
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=CONFIG["classes"]
    ))
    
    return all_preds, all_labels, all_confidences


def flag_low_confidence(sample_ids, predictions, confidences, threshold=0.70):
    """
    Flag predictions below confidence threshold for manual review.
    This implements the clinical safety requirement.
    """
    flagged = []
    for sid, pred, conf in zip(sample_ids, predictions, confidences):
        if conf < threshold:
            flagged.append({
                "sample_id": sid,
                "prediction": CONFIG["classes"][pred],
                "confidence": conf,
                "action": "FLAG FOR MANUAL CARDIOLOGIST REVIEW"
            })
    
    print(f"\n⚠️  {len(flagged)} samples flagged for manual review (confidence < {threshold})")
    return flagged


# =============================================================================
# Explainability
# =============================================================================

def generate_explanation(model, signal, prediction):
    """
    Generate an explanation for a model prediction.
    Implement your chosen explainability method here.
    
    Options:
    - Grad-CAM: highlights which time regions drove the prediction
    - SHAP: feature importance across leads and time steps
    - Attention weights: if using transformer architecture
    """
    # TODO: implement your explainability method
    raise NotImplementedError("Implement your explainability method")


def plot_ecg_with_explanation(signal, explanation, prediction, confidence):
    """
    Visualise the ECG signal overlaid with the explanation heatmap.
    """
    fig, axes = plt.subplots(12, 1, figsize=(20, 24))
    lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    
    for i, (ax, name) in enumerate(zip(axes, lead_names)):
        ax.plot(signal[i], color='blue', linewidth=0.8)
        ax.set_ylabel(name, rotation=0, labelpad=20)
        ax.set_xlim(0, len(signal[i]))
    
    fig.suptitle(
        f"Prediction: {CONFIG['classes'][prediction]} "
        f"(Confidence: {confidence:.1%})",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("ecg_explanation.png", dpi=150, bbox_inches='tight')
    print("Explanation saved to ecg_explanation.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n=== Starting ECG Experiment ===")
    print(f"Config: {CONFIG}")
    
    # TODO: Run your experiment:
    # 1. Load data
    # 2. Train / load model
    # 3. Evaluate on validation set
    # 4. Generate test predictions
    # 5. Flag low-confidence cases
    # 6. Generate explanations for predictions
    
    print("\n=== Experiment Complete ===")
    print("Remember to:")
    print("  1. Record results in benchmarking/method_comparison.md")
    print("  2. Update your individual workspace in members/")
    print("  3. Update this week's team journal in progress/")
    print("  4. Commit with a clear message:")
    print("     git commit -m 'Experiment: ResNet1D, PTB-XL, Accuracy=X%'")
