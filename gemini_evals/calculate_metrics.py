#!/usr/bin/env python3
"""
Script to calculate binary classification metrics from confusion matrices.
Calculates precision, recall, F1-score, support, and accuracy for each evaluation folder.
Saves results to CSV and creates visualizations.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def calculate_binary_metrics(confusion_matrix_path):
    """
    Calculate binary classification metrics from a confusion matrix CSV file.
    
    Args:
        confusion_matrix_path (str): Path to the confusion matrix CSV file
        
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    # Read confusion matrix
    cm_df = pd.read_csv(confusion_matrix_path, index_col=0)
    
    # Extract values from confusion matrix
    # Format: rows = true labels, columns = predicted labels
    tn = cm_df.iloc[0, 0]  # true_0, pred_0 (True Negatives)
    fp = cm_df.iloc[0, 1]  # true_0, pred_1 (False Positives)
    fn = cm_df.iloc[1, 0]  # true_1, pred_0 (False Negatives)
    tp = cm_df.iloc[1, 1]  # true_1, pred_1 (True Positives)
    
    # Calculate metrics
    total = tp + tn + fp + fn
    
    # Accuracy
    accuracy = (tp + tn) / total if total > 0 else 0
    
    # Precision for class 1 (positive class)
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Precision for class 0 (negative class)
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Recall for class 1 (positive class)
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Recall for class 0 (negative class)
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # F1-score for class 1
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    
    # F1-score for class 0
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    
    # Support (number of samples for each class)
    support_0 = tn + fp  # true class 0
    support_1 = tp + fn  # true class 1
    
    # Macro averages
    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2
    
    # Weighted averages
    weighted_precision = (precision_0 * support_0 + precision_1 * support_1) / total
    weighted_recall = (recall_0 * support_0 + recall_1 * support_1) / total
    weighted_f1 = (f1_0 * support_0 + f1_1 * support_1) / total
    
    return {
        'confusion_matrix': {
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        },
        'class_0': {
            'precision': precision_0,
            'recall': recall_0,
            'f1_score': f1_0,
            'support': support_0
        },
        'class_1': {
            'precision': precision_1,
            'recall': recall_1,
            'f1_score': f1_1,
            'support': support_1
        },
        'overall': {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'total_samples': total
        }
    }

def save_metrics_to_csv(all_results, output_path):
    """
    Save all metrics to a CSV file.
    
    Args:
        all_results (dict): Dictionary containing metrics for all folders
        output_path (str): Path to save the CSV file
    """
    # Create a list to store all rows
    rows = []
    
    for folder_name, metrics in all_results.items():
        # Overall metrics row
        overall_row = {
            'Dataset': folder_name,
            'Metric_Type': 'Overall',
            'Class': 'All',
            'Precision': metrics['overall']['weighted_precision'],
            'Recall': metrics['overall']['weighted_recall'],
            'F1_Score': metrics['overall']['weighted_f1'],
            'Support': metrics['overall']['total_samples'],
            'Accuracy': metrics['overall']['accuracy'],
            'Macro_Precision': metrics['overall']['macro_precision'],
            'Macro_Recall': metrics['overall']['macro_recall'],
            'Macro_F1': metrics['overall']['macro_f1']
        }
        rows.append(overall_row)
        
        # Class-specific metrics
        for class_num in ['0', '1']:
            class_key = f'class_{class_num}'
            class_row = {
                'Dataset': folder_name,
                'Metric_Type': 'Per_Class',
                'Class': class_num,
                'Precision': metrics[class_key]['precision'],
                'Recall': metrics[class_key]['recall'],
                'F1_Score': metrics[class_key]['f1_score'],
                'Support': metrics[class_key]['support'],
                'Accuracy': metrics['overall']['accuracy'],  # Same for all classes
                'Macro_Precision': metrics['overall']['macro_precision'],
                'Macro_Recall': metrics['overall']['macro_recall'],
                'Macro_F1': metrics['overall']['macro_f1']
            }
            rows.append(class_row)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nMetrics saved to: {output_path}")

def create_visualizations(all_results, output_dir):
    """
    Create and save visualizations of the metrics.
    
    Args:
        all_results (dict): Dictionary containing metrics for all folders
        output_dir (str): Directory to save the plots
    """
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Binary Classification Metrics Comparison', fontsize=16, fontweight='bold')
    
    datasets = list(all_results.keys())
    metrics_to_plot = [
        ('Accuracy', 'overall', 'accuracy'),
        ('Macro F1-Score', 'overall', 'macro_f1'),
        ('Weighted F1-Score', 'overall', 'weighted_f1'),
        ('Macro Precision', 'overall', 'macro_precision')
    ]
    
    for i, (metric_name, category, metric_key) in enumerate(metrics_to_plot):
        ax = axes[i//2, i%2]
        values = [all_results[dataset][category][metric_key] for dataset in datasets]
        
        bars = ax.bar(datasets, values, color=['#3498db', '#e74c3c'], alpha=0.8)
        ax.set_title(metric_name, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-Class Metrics Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Per-Class Metrics Comparison', fontsize=16, fontweight='bold')
    
    class_metrics = ['precision', 'recall', 'f1_score']
    metric_names = ['Precision', 'Recall', 'F1-Score']
    
    for i, (metric, name) in enumerate(zip(class_metrics, metric_names)):
        ax = axes[i]
        
        # Data for both classes and datasets
        x = np.arange(len(datasets))
        width = 0.35
        
        class_0_values = [all_results[dataset]['class_0'][metric] for dataset in datasets]
        class_1_values = [all_results[dataset]['class_1'][metric] for dataset in datasets]
        
        bars1 = ax.bar(x - width/2, class_0_values, width, label='Class 0', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, class_1_values, width, label='Class 1', color='#e74c3c', alpha=0.8)
        
        ax.set_title(name, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xlabel('Dataset')
        ax.set_xticks(x)
        ax.set_xticklabels([d for d in datasets])
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_class_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
    
    for i, (dataset, metrics) in enumerate(all_results.items()):
        ax = axes[i]
        cm = metrics['confusion_matrix']
        
        # Create confusion matrix array
        cm_array = np.array([[cm['tn'], cm['fp']], 
                            [cm['fn'], cm['tp']]])
        
        # Create heatmap
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'])
        
        # ax.set_title(dataset.replace(' (Validation Set Only)', '\n(Validation Set Only)'), 
                    # fontweight='bold')
        
        # Add accuracy text
        accuracy = metrics['overall']['accuracy']
        ax.text(0.5, -0.15, f'Accuracy: {accuracy:.3f}', 
               transform=ax.transAxes, ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Support (Sample Size) Comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    class_0_support = [all_results[dataset]['class_0']['support'] for dataset in datasets]
    class_1_support = [all_results[dataset]['class_1']['support'] for dataset in datasets]
    
    bars1 = ax.bar(x - width/2, class_0_support, width, label='Class 0', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, class_1_support, width, label='Class 1', color='#e74c3c', alpha=0.8)
    
    ax.set_title('Sample Size Comparison by Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels([d for d in datasets])
    ax.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sample_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to: {output_dir}")
    print("Created files:")
    print("  - overall_metrics_comparison.png")
    print("  - per_class_metrics_comparison.png") 
    print("  - confusion_matrices.png")
    print("  - sample_size_comparison.png")

def print_metrics(metrics, folder_name):
    """
    Print formatted metrics for a given evaluation folder.
    
    Args:
        metrics (dict): Metrics dictionary from calculate_binary_metrics
        folder_name (str): Name of the evaluation folder
    """
    print(f"\n{'='*60}")
    print(f"METRICS FOR: {folder_name}")
    print(f"{'='*60}")
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"               0      1")
    print(f"Actual    0   {cm['tn']:4d}   {cm['fp']:4d}")
    print(f"          1   {cm['fn']:4d}   {cm['tp']:4d}")
    
    # Overall metrics
    overall = metrics['overall']
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:           {overall['accuracy']:.4f}")
    print(f"  Total Samples:      {overall['total_samples']}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print(f"{'-'*50}")
    
    class_0 = metrics['class_0']
    class_1 = metrics['class_1']
    
    print(f"{'0':<8} {class_0['precision']:<10.4f} {class_0['recall']:<10.4f} "
          f"{class_0['f1_score']:<10.4f} {class_0['support']:<8}")
    print(f"{'1':<8} {class_1['precision']:<10.4f} {class_1['recall']:<10.4f} "
          f"{class_1['f1_score']:<10.4f} {class_1['support']:<8}")
    
    # Averaged metrics
    print(f"\nAveraged Metrics:")
    print(f"  Macro Precision:    {overall['macro_precision']:.4f}")
    print(f"  Macro Recall:       {overall['macro_recall']:.4f}")
    print(f"  Macro F1-Score:     {overall['macro_f1']:.4f}")
    print(f"  Weighted Precision: {overall['weighted_precision']:.4f}")
    print(f"  Weighted Recall:    {overall['weighted_recall']:.4f}")
    print(f"  Weighted F1-Score:  {overall['weighted_f1']:.4f}")

def main():
    """
    Main function to process both evaluation folders and calculate metrics.
    """
    # Define the paths to the evaluation folders
    base_path = Path("/Users/derekaskaryar/code for classes/cs152/gemini_evals")
    
    folders = [
        {
            'path': base_path / "gemini_binary_bot_data_eval_outputs",
            'name': "Bot Data Evaluation"
        },
        {
            'path': base_path / "gemini_binary_user_data_eval_outputs_val_set_only",
            'name': "User Data Evaluation"
        }
    ]
    
    print("Binary Classification Metrics Calculator")
    print("=" * 60)
    
    all_results = {}
    
    for folder_info in folders:
        folder_path = folder_info['path']
        folder_name = folder_info['name']
        confusion_matrix_path = folder_path / "confusion_matrix.csv"
        
        if confusion_matrix_path.exists():
            try:
                metrics = calculate_binary_metrics(confusion_matrix_path)
                all_results[folder_name] = metrics
                print_metrics(metrics, folder_name)
            except Exception as e:
                print(f"\nError processing {folder_name}: {e}")
        else:
            print(f"\nWarning: Confusion matrix not found at {confusion_matrix_path}")
    
    # Summary comparison
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        
        print(f"{'Metric':<25} {'Bot Data':<15} {'User Data':<15}")
        print(f"{'-'*55}")
        
        for folder_name, metrics in all_results.items():
            if folder_name == list(all_results.keys())[0]:  # First folder (Bot Data)
                bot_metrics = metrics
            else:  # Second folder (User Data)
                user_metrics = metrics
        
        if 'bot_metrics' in locals() and 'user_metrics' in locals():
            comparisons = [
                ('Accuracy', 'overall', 'accuracy'),
                ('Macro F1-Score', 'overall', 'macro_f1'),
                ('Weighted F1-Score', 'overall', 'weighted_f1'),
                ('Class 1 Precision', 'class_1', 'precision'),
                ('Class 1 Recall', 'class_1', 'recall'),
                ('Class 1 F1-Score', 'class_1', 'f1_score')
            ]
            
            for metric_name, category, metric_key in comparisons:
                bot_val = bot_metrics[category][metric_key]
                user_val = user_metrics[category][metric_key]
                print(f"{metric_name:<25} {bot_val:<15.4f} {user_val:<15.4f}")
    
    # Save results to CSV
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_output_path = base_path / f"metrics_results_{timestamp}.csv"
        save_metrics_to_csv(all_results, csv_output_path)
        
        # Create visualizations
        viz_output_dir = base_path / f"visualizations_{timestamp}"
        create_visualizations(all_results, viz_output_dir)

if __name__ == "__main__":
    main() 