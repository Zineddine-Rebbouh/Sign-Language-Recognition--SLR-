"""
Model comparison script for Sign Language Recognition.
Compares all trained models and generates summary reports.

Run from the project root:
    python scripts/compare_models.py --results_dir ./results/metrics --output_dir ./results/figures
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
import matplotlib
matplotlib.use("Agg")  # headless — safe for servers without a display
import matplotlib.pyplot as plt
import seaborn as sns

# ── Bootstrap sys.path so `src` imports work from any directory ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> List[Dict]:
    """
    Load all result JSON files from a directory.
    
    Args:
        results_dir: Directory containing result JSON files
    
    Returns:
        List of result dictionaries
    """
    results = []
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return results
    
    # Find all JSON files
    json_files = list(results_dir.rglob("*_results.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    return results


def compare_models(
    results: List[Dict],
    save_path: Optional[Path] = None,
    save_plot: Optional[Path] = None
) -> pd.DataFrame:
    """
    Compare multiple models and generate summary.
    
    Args:
        results: List of result dictionaries
        save_path: Optional path to save comparison CSV
        save_plot: Optional path to save comparison plot
    
    Returns:
        DataFrame with comparison results
    """
    if len(results) == 0:
        logger.warning("No results to compare")
        return pd.DataFrame()
    
    # Extract metrics
    comparison_data = {
        'Model': [],
        'Accuracy': [],
        'Top-5 Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    
    for result in results:
        model_name = result.get('model_name', 'Unknown')
        comparison_data['Model'].append(model_name)
        comparison_data['Accuracy'].append(result.get('accuracy', 0.0))
        comparison_data['Top-5 Accuracy'].append(
            result.get('top_k_accuracy', {}).get('top_5_accuracy', 
            result.get('top_5_accuracy', 0.0))
        )
        comparison_data['Precision'].append(result.get('precision', 0.0))
        comparison_data['Recall'].append(result.get('recall', 0.0))
        comparison_data['F1-Score'].append(result.get('f1_score', 0.0))
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Accuracy', ascending=False)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*80)
    logger.info(f"\n{df.to_string(index=False)}")
    
    # Save CSV
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"\nComparison saved to: {save_path}")
    
    # Create visualization
    if save_plot and len(df) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Top-5 Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            df_sorted = df.sort_values(metric, ascending=True)
            ax.barh(df_sorted['Model'], df_sorted[metric])
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.grid(axis='x', alpha=0.3)
        
        # Remove empty subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        save_plot = Path(save_plot)
        save_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to: {save_plot}")
        plt.close()
    
    return df


def main():
    """Main function to compare models."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Sign Language Recognition models')
    parser.add_argument('--results_dir', type=str, default='/kaggle/working/slr_results',
                       help='Directory containing result JSON files')
    parser.add_argument('--output', type=str, default='/kaggle/working/model_comparison.csv',
                       help='Output CSV file path')
    parser.add_argument('--plot', type=str, default='/kaggle/working/model_comparison.png',
                       help='Output plot file path')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(Path(args.results_dir))
    
    if len(results) == 0:
        logger.error("No results found. Please check the results directory.")
        return
    
    # Compare
    df = compare_models(
        results,
        save_path=Path(args.output),
        save_plot=Path(args.plot)
    )
    
    logger.info(f"\n✓ Comparison complete. Found {len(results)} models.")


if __name__ == "__main__":
    main()

