"""
Visualization utilities for model evaluation and analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve, auc,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import learning_curve


# ============ General Plotting Functions ============

def plot_confusion_matrix(y_true, y_pred, save_path, split_name=""):
    """Plot and save confusion matrix as PNG"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Opened', 'Opened'],
                yticklabels=['Not Opened', 'Opened'])
    plt.title(f'Confusion Matrix - {split_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def plot_learning_curve(model, X, y, save_path, cv=5):
    """Plot learning curve showing train/validation scores vs training size"""
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    plt.plot(train_sizes, valid_mean, 'o-', color='orange', label='Cross-Validation Score')
    plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, 
                     alpha=0.1, color='orange')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score')
    plt.title('Learning Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Learning curve saved to {save_path}")


def plot_loss_learning_curve(model, X, y, save_path, cv=5):
    """Plot learning curve showing train/validation log loss vs training size"""
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        scoring='neg_log_loss',
        n_jobs=-1
    )
    
    # neg_log_loss returns negative values, convert to positive for plotting
    train_loss = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_loss = -np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_loss, 'o-', color='blue', label='Training Loss')
    plt.fill_between(train_sizes, train_loss - train_std, train_loss + train_std, 
                     alpha=0.1, color='blue')
    plt.plot(train_sizes, valid_loss, 'o-', color='orange', label='Cross-Validation Loss')
    plt.fill_between(train_sizes, valid_loss - valid_std, valid_loss + valid_std, 
                     alpha=0.1, color='orange')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Log Loss')
    plt.title('Learning Curve - Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Loss learning curve saved to {save_path}")


def plot_pr_curve(y_true, y_proba, save_path, split_name=""):
    """Plot Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.fill_between(recall, precision, alpha=0.2, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {split_name}')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"PR curve saved to {save_path}")
    return pr_auc


def plot_roc_curve(y_true, y_proba, save_path, split_name=""):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    plt.fill_between(fpr, tpr, alpha=0.2, color='blue')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {split_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"ROC curve saved to {save_path}")
    return roc_auc


def plot_coefficients(model, feature_names, save_path):
    """Plot logistic regression coefficients as horizontal bar chart"""
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=True)
    
    colors = ['green' if c > 0 else 'red' for c in coef_df["coefficient"]]
    
    plt.figure(figsize=(10, 8))
    plt.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Logistic Regression Coefficients')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Coefficients plot saved to {save_path}")
    return coef_df


# ============ LightGBM-Specific Plotting Functions ============

def plot_feature_importance(model, feature_names, save_path):
    """Plot LightGBM feature importance as horizontal bar chart"""
    importance = model.feature_importances_
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df["feature"], importance_df["importance"], color='steelblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('LightGBM Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Feature importance plot saved to {save_path}")
    return importance_df


def plot_training_history(evals_result, save_path):
    """Plot training and validation loss curves over boosting rounds"""
    train_loss = evals_result['training']['binary_logloss']
    valid_loss = evals_result['valid_1']['binary_logloss']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(valid_loss, label='Validation Loss', color='orange')
    plt.xlabel('Boosting Round')
    plt.ylabel('Binary Log Loss')
    plt.title('LightGBM Training History - Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Training history saved to {save_path}")


def plot_accuracy_curves(evals_result, save_path):
    """Plot training and validation accuracy curves over boosting rounds"""
    # binary_error is 1 - accuracy, so accuracy = 1 - binary_error
    train_error = evals_result['training']['binary_error']
    valid_error = evals_result['valid_1']['binary_error']
    
    train_acc = [1 - e for e in train_error]
    valid_acc = [1 - e for e in valid_error]
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Training Accuracy', color='blue')
    plt.plot(valid_acc, label='Validation Accuracy', color='orange')
    plt.xlabel('Boosting Round')
    plt.ylabel('Accuracy')
    plt.title('LightGBM Training History - Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Accuracy curves saved to {save_path}")


def compute_metrics_per_round(model, X_train, y_train, X_valid, y_valid):
    """
    Compute F1, precision, and recall at each boosting round for train and validation.
    Returns dict with metrics arrays for plotting.
    """
    n_rounds = model.n_estimators_
    
    metrics = {
        'train_f1': [], 'valid_f1': [],
        'train_precision': [], 'valid_precision': [],
        'train_recall': [], 'valid_recall': []
    }
    
    for i in range(1, n_rounds + 1):
        # Predict using first i trees
        y_train_pred = (model.predict_proba(X_train, num_iteration=i)[:, 1] >= 0.5).astype(int)
        y_valid_pred = (model.predict_proba(X_valid, num_iteration=i)[:, 1] >= 0.5).astype(int)
        
        # Compute metrics
        metrics['train_f1'].append(f1_score(y_train, y_train_pred, zero_division=0))
        metrics['valid_f1'].append(f1_score(y_valid, y_valid_pred, zero_division=0))
        metrics['train_precision'].append(precision_score(y_train, y_train_pred, zero_division=0))
        metrics['valid_precision'].append(precision_score(y_valid, y_valid_pred, zero_division=0))
        metrics['train_recall'].append(recall_score(y_train, y_train_pred, zero_division=0))
        metrics['valid_recall'].append(recall_score(y_valid, y_valid_pred, zero_division=0))
    
    return metrics


def plot_f1_curves(metrics, save_path):
    """Plot training and validation F1 score curves over boosting rounds"""
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_f1'], label='Training F1', color='blue')
    plt.plot(metrics['valid_f1'], label='Validation F1', color='orange')
    plt.xlabel('Boosting Round')
    plt.ylabel('F1 Score')
    plt.title('LightGBM Training History - F1 Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"F1 curves saved to {save_path}")


def plot_precision_curves(metrics, save_path):
    """Plot training and validation precision curves over boosting rounds"""
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_precision'], label='Training Precision', color='blue')
    plt.plot(metrics['valid_precision'], label='Validation Precision', color='orange')
    plt.xlabel('Boosting Round')
    plt.ylabel('Precision')
    plt.title('LightGBM Training History - Precision')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Precision curves saved to {save_path}")


def plot_recall_curves(metrics, save_path):
    """Plot training and validation recall curves over boosting rounds"""
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_recall'], label='Training Recall', color='blue')
    plt.plot(metrics['valid_recall'], label='Validation Recall', color='orange')
    plt.xlabel('Boosting Round')
    plt.ylabel('Recall')
    plt.title('LightGBM Training History - Recall')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Recall curves saved to {save_path}")


def plot_loss_learning_curve_lgbm(X_train, y_train, X_valid, y_valid, hyperparams, save_path, n_sizes=10):
    """
    Plot learning curve showing train/validation log loss vs training set size for LightGBM.
    Trains multiple models on different training set sizes.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_valid: Validation features
        y_valid: Validation labels
        hyperparams: LightGBM hyperparameters dict
        save_path: Path to save the plot
        n_sizes: Number of training sizes to evaluate
    """
    from lightgbm import LGBMClassifier
    from sklearn.metrics import log_loss
    
    train_sizes_ratio = np.linspace(0.1, 1.0, n_sizes)
    train_sizes = (train_sizes_ratio * len(X_train)).astype(int)
    
    train_losses = []
    valid_losses = []
    
    # Remove early_stopping_rounds from hyperparams for this function
    hyperparams_copy = hyperparams.copy()
    hyperparams_copy.pop("early_stopping_rounds", None)
    
    print(f"Computing loss learning curve across {n_sizes} training sizes...")
    
    for size in train_sizes:
        # Sample training data
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]
        
        # Train model
        model = LGBMClassifier(**hyperparams_copy, verbose=-1)
        model.fit(X_train_subset, y_train_subset)
        
        # Compute losses
        y_train_proba = model.predict_proba(X_train_subset)[:, 1]
        y_valid_proba = model.predict_proba(X_valid)[:, 1]
        
        train_losses.append(log_loss(y_train_subset, y_train_proba))
        valid_losses.append(log_loss(y_valid, y_valid_proba))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_losses, 'o-', color='blue', label='Training Loss')
    plt.plot(train_sizes, valid_losses, 'o-', color='orange', label='Validation Loss')
    plt.xlabel('Training Set Size')
    plt.ylabel('Log Loss')
    plt.title('LightGBM Learning Curve - Loss vs Training Size')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Loss learning curve saved to {save_path}")
