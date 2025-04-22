import os
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import glob
import cv2

parser = argparse.ArgumentParser(description='Evaluate depth prediction results from already completed inference')
parser.add_argument('--original_data_path', type=str, help='Path to original endoscopic dataset')
parser.add_argument('--results_path', type=str, help='Path to inference results directory')
parser.add_argument('--output-dir', type=str, default='evaluation_metrics', help='Output directory for metrics')
args = parser.parse_args()

def compute_metrics(pred_depth, gt_depth):
    """Compute various depth evaluation metrics"""
    # Debug: Check shapes
    print(f"Pred shape: {pred_depth.shape}, GT shape: {gt_depth.shape}")
    
    # Ensure both are 2D
    if len(pred_depth.shape) > 2:
        pred_depth = pred_depth.squeeze()
    if len(gt_depth.shape) > 2:
        gt_depth = gt_depth.squeeze()
    
    # Resize if shapes don't match
    if pred_depth.shape != gt_depth.shape:
        print(f"Resizing predicted depth from {pred_depth.shape} to {gt_depth.shape}")
        pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Filter valid depth values
    min_depth = 0.0001
    max_depth = 1000.0  # Increased max depth to be more inclusive
    
    # Check for valid depth values
    valid_mask = (gt_depth > min_depth) & (gt_depth < max_depth) & (gt_depth != 0)
    
    # Debug: Print mask statistics
    print(f"Valid mask: {valid_mask.sum()} out of {valid_mask.size} pixels")
    
    # Only evaluate where we have valid ground truth
    if valid_mask.sum() < 100:  # Minimum number of valid pixels
        print(f"Warning: Not enough valid pixels ({valid_mask.sum()})")
        return None
        
    pred_depth_valid = pred_depth[valid_mask]
    gt_depth_valid = gt_depth[valid_mask]
    
    # Check if we have reasonable values
    if np.all(pred_depth_valid == 0) or np.all(gt_depth_valid == 0):
        print("Warning: All values are zero")
        return None
    
    # Scale prediction to match ground truth
    # Use robust scale estimation to handle outliers
    scale = np.median(gt_depth_valid) / np.median(pred_depth_valid)
    
    # Debug: Print scale factor
    print(f"Scale factor: {scale}")
    
    pred_depth_scaled = pred_depth_valid * scale
    
    # Compute metrics
    rmse = np.sqrt(np.mean((pred_depth_scaled - gt_depth_valid) ** 2))
    abs_rel = np.mean(np.abs(pred_depth_scaled - gt_depth_valid) / gt_depth_valid)
    sq_rel = np.mean(((pred_depth_scaled - gt_depth_valid) ** 2) / gt_depth_valid)
    
    # Handle log metrics carefully
    valid_log_mask = (pred_depth_scaled > 0) & (gt_depth_valid > 0)
    if valid_log_mask.sum() > 0:
        rmse_log = np.sqrt(np.mean((np.log(pred_depth_scaled[valid_log_mask]) - np.log(gt_depth_valid[valid_log_mask])) ** 2))
    else:
        rmse_log = float('inf')
    
    threshold_ratios = np.maximum(pred_depth_scaled / gt_depth_valid, gt_depth_valid / pred_depth_scaled)
    delta1 = np.mean(threshold_ratios < 1.25) * 100
    delta2 = np.mean(threshold_ratios < 1.25 ** 2) * 100
    delta3 = np.mean(threshold_ratios < 1.25 ** 3) * 100
    
    metrics = {
        'RMSE': rmse,
        'Abs Rel': abs_rel,
        'Sq Rel': sq_rel,
        'RMSE log': rmse_log,
        'δ < 1.25': delta1,
        'δ < 1.25²': delta2,
        'δ < 1.25³': delta3,
    }
    
    return metrics

def visualize_depth(depth):
    """Create color visualization of depth map"""
    depth = depth.squeeze()
    
    # Ignore zero values in statistics
    valid_mask = depth > 0
    if valid_mask.sum() > 0:
        depth_min = depth[valid_mask].min()
        depth_max = depth[valid_mask].max()
    else:
        depth_min = 0
        depth_max = 1
    
    if depth_max - depth_min < 1e-6:
        normalized_depth = np.zeros_like(depth)
    else:
        normalized_depth = (depth - depth_min) / (depth_max - depth_min)
    
    # Apply mask for better visualization
    if valid_mask.sum() > 0:
        normalized_depth[~valid_mask] = 0
    
    colored_depth = (plt.cm.plasma(normalized_depth)[:, :, :3] * 255).astype(np.uint8)
    
    # Make zero values black
    colored_depth[~valid_mask] = 0
    
    return colored_depth

def load_ground_truth_depth(organ_dir, image_id):
    """Load ground truth depth from the original dataset"""
    gt_depth_path = organ_dir / 'depths' / f"{image_id}.npy"
    if not gt_depth_path.exists():
        return None
    return np.load(gt_depth_path).astype(np.float32)

def load_predicted_depth(results_organ_dir, image_id):
    """Load predicted depth from inference results"""
    pred_depth_path = results_organ_dir / 'depth' / f"{image_id}_depth.npy"
    if not pred_depth_path.exists():
        return None
    return np.load(pred_depth_path).astype(np.float32)

def create_comparison_visualization(image_path, pred_depth, gt_depth, output_dir, idx):
    """Create visualization comparing predicted and ground truth depth"""
    vis_dir = output_dir / 'visualizations'
    vis_dir.makedirs_p()
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return
    
    # Resize image to match depth maps
    img = cv2.resize(img, (512, 288))  # Using the dimensions from your inference script
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Ensure depths match the expected dimensions
    if pred_depth.shape != gt_depth.shape:
        pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Create depth visualizations
    pred_viz = visualize_depth(pred_depth)
    gt_viz = visualize_depth(gt_depth)
    
    # Calculate error map
    error = np.abs(pred_depth - gt_depth)
    error_viz = visualize_depth(error)
    
    # Create side-by-side comparison
    vis = np.hstack([img, pred_viz, gt_viz, error_viz])
    
    # Add labels
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    ax.imshow(vis)
    ax.set_title(f'RGB | Predicted | Ground Truth | Error - Sample {idx}')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(str(vis_dir / f'comparison_{idx}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save individual depth maps for debugging
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pred_depth)
    axes[0].set_title(f'Predicted Depth\nMin: {pred_depth.min():.3f}, Max: {pred_depth.max():.3f}')
    axes[1].imshow(gt_depth)
    axes[1].set_title(f'Ground Truth Depth\nMin: {gt_depth.min():.3f}, Max: {gt_depth.max():.3f}')
    axes[2].imshow(error)
    axes[2].set_title(f'Error\nMin: {error.min():.3f}, Max: {error.max():.3f}')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(str(vis_dir / f'debug_{idx}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_results():
    """Evaluate the inference results against ground truth"""
    original_data_path = Path(args.original_data_path)
    results_path = Path(args.results_path)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()
    
    # Get all organ directories from the original dataset
    organ_dirs = [d for d in original_data_path.iterdir() if d.is_dir()]
    
    all_metrics = []
    sample_count = 0
    visualized_samples = 0
    failed_samples = 0
    
    print(f"Evaluating results for {len(organ_dirs)} organs...")
    
    for organ_dir in organ_dirs:
        organ_name = organ_dir.name
        results_organ_dir = results_path / organ_name
        
        if not results_organ_dir.exists():
            print(f"Warning: No results found for {organ_name}")
            continue
            
        # Get all images in the original dataset for this organ
        image_dir = organ_dir / 'images'
        image_files = sorted(glob.glob(str(image_dir / '*.png')))
        
        print(f"Processing {organ_name} with {len(image_files)} images...")
        
        for image_path in tqdm(image_files):
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            
            # Load ground truth depth
            gt_depth = load_ground_truth_depth(organ_dir, image_id)
            if gt_depth is None:
                print(f"No GT depth for {image_id}")
                continue
                
            # Load predicted depth
            pred_depth = load_predicted_depth(results_organ_dir, image_id)
            if pred_depth is None:
                print(f"No predicted depth for {image_id}")
                continue
            
            # Debug: Print shapes and statistics
            print(f"\nProcessing {image_id}")
            print(f"GT shape: {gt_depth.shape}, Pred shape: {pred_depth.shape}")
            print(f"GT range: [{gt_depth.min()}, {gt_depth.max()}]")
            print(f"Pred range: [{pred_depth.min()}, {pred_depth.max()}]")
            
            # Compute metrics for this sample
            metrics = compute_metrics(pred_depth, gt_depth)
            
            if metrics is not None:
                all_metrics.append(metrics)
                sample_count += 1
                
                # Create visualization for first few samples
                if visualized_samples < 10:
                    create_comparison_visualization(image_path, pred_depth, gt_depth, output_dir, visualized_samples)
                    visualized_samples += 1
            else:
                failed_samples += 1
    
    print(f"\nProcessed {sample_count} samples successfully, {failed_samples} failed")
    
    # Aggregate metrics across all samples
    if len(all_metrics) > 0:
        avg_metrics = {
            k: np.mean([m[k] for m in all_metrics])
            for k in all_metrics[0].keys()
        }
        
        # Convert numpy types to Python types for JSON serialization
        avg_metrics_serializable = {
            k: float(v) for k, v in avg_metrics.items()
        }
        
        # Save metrics to JSON
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(avg_metrics_serializable, f, indent=4)
        
        # Print metrics
        print(f"\nEvaluation Results (based on {sample_count} samples):")
        print("-" * 40)
        for metric, value in avg_metrics.items():
            print(f"{metric:10}: {float(value):.4f}")
        print("-" * 40)
        
        # Create and save a metrics plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.bar(range(len(avg_metrics)), list(avg_metrics.values()))
        ax.set_xticks(range(len(avg_metrics)))
        ax.set_xticklabels(list(avg_metrics.keys()), rotation=45)
        ax.set_ylabel('Metric Value')
        ax.set_title('Depth Prediction Evaluation Metrics')
        plt.tight_layout()
        plt.savefig(str(output_dir / 'metrics_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nResults saved to {output_dir}")
    else:
        print("Evaluation failed - no valid predictions found")

if __name__ == "__main__":
    evaluate_results()