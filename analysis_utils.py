"""
Qualitative analysis utilities for model evaluation.
"""
import pandas as pd
import numpy as np
import os


def export_confusion_matrix_splits(y_true, y_pred, data_df, save_dir, prefix="valid"):
    """
    Split validation data into TP, TN, FP, FN CSVs for qualitative analysis.
    """
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Reset index to ensure alignment
    data_df = data_df.reset_index(drop=True)
    
    # Create masks for each category
    tp_mask = (y_true == 1) & (y_pred == 1)  # True Positives
    tn_mask = (y_true == 0) & (y_pred == 0)  # True Negatives
    fp_mask = (y_true == 0) & (y_pred == 1)  # False Positives
    fn_mask = (y_true == 1) & (y_pred == 0)  # False Negatives
    
    # Create DataFrames for each category
    tp_df = data_df[tp_mask].copy()
    tn_df = data_df[tn_mask].copy()
    fp_df = data_df[fp_mask].copy()
    fn_df = data_df[fn_mask].copy()
    
    # Add prediction info columns
    tp_df['y_true'] = 1
    tp_df['y_pred'] = 1
    tp_df['category'] = 'TP'
    
    tn_df['y_true'] = 0
    tn_df['y_pred'] = 0
    tn_df['category'] = 'TN'
    
    fp_df['y_true'] = 0
    fp_df['y_pred'] = 1
    fp_df['category'] = 'FP'
    
    fn_df['y_true'] = 1
    fn_df['y_pred'] = 0
    fn_df['category'] = 'FN'
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save to CSV files
    file_paths = {}
    
    tp_path = f"{save_dir}/{prefix}_TP.csv"
    tn_path = f"{save_dir}/{prefix}_TN.csv"
    fp_path = f"{save_dir}/{prefix}_FP.csv"
    fn_path = f"{save_dir}/{prefix}_FN.csv"
    
    tp_df.to_csv(tp_path, index=False)
    tn_df.to_csv(tn_path, index=False)
    fp_df.to_csv(fp_path, index=False)
    fn_df.to_csv(fn_path, index=False)
    
    file_paths = {
        'TP': tp_path,
        'TN': tn_path,
        'FP': fp_path,
        'FN': fn_path
    }
    
    # Print summary
    print(f"\nConfusion Matrix Split Summary ({prefix}):")
    print(f"  True Positives (TP):  {len(tp_df)} samples -> {tp_path}")
    print(f"  True Negatives (TN):  {len(tn_df)} samples -> {tn_path}")
    print(f"  False Positives (FP): {len(fp_df)} samples -> {fp_path}")
    print(f"  False Negatives (FN): {len(fn_df)} samples -> {fn_path}")
    
    return file_paths
