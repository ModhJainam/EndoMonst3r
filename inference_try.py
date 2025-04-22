import os
import torch
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import tarfile
import tempfile

# Add import paths if needed
import sys
sys.path.append('.')

# Parse arguments
parser = argparse.ArgumentParser(description='Run inference with pretrained endoSFMLearner on Unity dataset')
parser.add_argument('--data_path', type=str, help='Path to Unity camera dataset')
parser.add_argument('--weights-path', type=str, required=True, help='Path to folder with .tar weight files')
parser.add_argument('--output-dir', type=str, default='inference_results', help='Output directory')
parser.add_argument('--img-height', type=int, default=288, help='Image height')
parser.add_argument('--img-width', type=int, default=512, help='Image width')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda')
args = parser.parse_args()

# Use CPU if no CUDA available or requested
device = torch.device("cuda") if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
print(f"Using device: {device}")

def extract_tar_to_temp(tar_path):
    """Extract a tar file to a temporary directory and return the path"""
    temp_dir = tempfile.mkdtemp()
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=temp_dir)
    return temp_dir

def load_pretrained_models():
    """Load pretrained endoSfMLearner models from tar files"""
    print("Loading pretrained models from", args.weights_path)
    
    # Import model modules based on the actual structure
    from models.resnet_encoder import ResnetEncoder
    from models.DispResNet import DispResNet
    from models.PoseResNet import PoseResNet
    
    # Create model instances
    disp_net = DispResNet(18, False)
    pose_net = PoseResNet(18, False)
    
    # Move models to device
    disp_net.to(device)
    pose_net.to(device)
    
    # Load weights from tar files
    weights_path = Path(args.weights_path)
    
    # Load depth model (dispnet)
    dispnet_path = weights_path / "dispnet_model_best.pth.tar"
    if dispnet_path.exists():
        print(f"Loading depth model from {dispnet_path}")
        try:
            # Load from tar without extracting
            checkpoint = torch.load(dispnet_path, map_location=device)
            if 'state_dict' in checkpoint:
                # Convert state dict to float
                float_state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    float_state_dict[k] = v.float()
                
                disp_net.load_state_dict(float_state_dict)
                print("Depth model loaded successfully")
            else:
                print("Warning: 'state_dict' not found in checkpoint")
        except Exception as e:
            print(f"Error loading depth model: {e}")
    else:
        print(f"Warning: Depth model not found at {dispnet_path}")
    
    # Load pose model
    pose_path = weights_path / "exp_pose_model_best.pth.tar"
    if pose_path.exists():
        print(f"Loading pose model from {pose_path}")
        try:
            # Load from tar without extracting
            checkpoint = torch.load(pose_path, map_location=device)
            if 'state_dict' in checkpoint:
                # Convert state dict to float
                float_state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    float_state_dict[k] = v.float()
                
                pose_net.load_state_dict(float_state_dict)
                print("Pose model loaded successfully")
            else:
                print("Warning: 'state_dict' not found in checkpoint")
        except Exception as e:
            print(f"Error loading pose model: {e}")
    else:
        print(f"Warning: Pose model not found at {pose_path}")
    
    # Ensure all model parameters are float32
    for param in disp_net.parameters():
        param.data = param.data.float()
    
    for param in pose_net.parameters():
        param.data = param.data.float()
    
    # Set models to evaluation mode
    disp_net.eval()
    pose_net.eval()
    
    # Print model data types to confirm
    for name, param in disp_net.named_parameters():
        print(f"Disp net parameter {name} is {param.dtype}")
        break
    
    return disp_net, pose_net

def preprocess_image(img):
    """Preprocess image for the network"""
    img = cv2.resize(img, (args.img_width, args.img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # Normalize using ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Subtract mean and divide by std
    img = (img - mean) / std
    
    # Convert to tensor [B, C, H, W] with float32 precision
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Double-check dtype
    print(f"Input tensor dtype: {img.dtype}")
    
    return img
@torch.no_grad()
def process_directory(organ_path, disp_net, pose_net):
    """Process all images in an organ directory"""
    output_dir = Path(args.output_dir) / organ_path.name
    output_dir.makedirs_p()
    
    # Get all PNG images in the images folder
    image_dir = organ_path / 'images'
    image_files = sorted(image_dir.files('*.png'))
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Processing {len(image_files)} images from {organ_path.name}...")
    
    # Create results folders
    depth_dir = output_dir / 'depth'
    pose_dir = output_dir / 'poses'
    vis_dir = output_dir / 'visualizations'
    
    depth_dir.makedirs_p()
    pose_dir.makedirs_p()
    vis_dir.makedirs_p()
    
    # Process images
    for i in tqdm(range(len(image_files) - 1)):
        try:
            # Get consecutive frames
            img1_path = image_files[i]
            img2_path = image_files[i + 1]
            
            # Read and preprocess images
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            if img1 is None or img2 is None:
                print(f"Failed to load image: {img1_path} or {img2_path}")
                continue
                
            img1_tensor = preprocess_image(img1)
            img2_tensor = preprocess_image(img2)
            
            # Compute depth
            disp = disp_net(img1_tensor)  # This should return a disparity map
            
            # Convert disparity to depth (1/disp)
            if isinstance(disp, (list, tuple)):
                disp = disp[0]  # Take the first element if it's a list or tuple
                
            depth_np = 1.0 / (disp.squeeze().cpu().numpy() + 1e-6)
            
            # Compute pose (from img1 to img2)
            pose = pose_net(img1_tensor, img2_tensor)  # This should return a pose matrix or parameters
            
            # Process pose output based on its format
            if isinstance(pose, (list, tuple)):
                # If it's a tuple, it might be (rotation, translation)
                if len(pose) == 2:
                    rotation, translation = pose
                    pose_mat = np.eye(4)
                    pose_mat[:3, :3] = rotation.squeeze().cpu().numpy()
                    pose_mat[:3, 3] = translation.squeeze().cpu().numpy()
                else:
                    pose_mat = pose[0].squeeze().cpu().numpy()
            else:
                # Assume it's already a matrix
                pose_mat = pose.squeeze().cpu().numpy()
            
            # Save results
            img_name = os.path.basename(img1_path).split('.')[0]
            
            # Save depth
            np.save(str(depth_dir / f"{img_name}_depth.npy"), depth_np)
            
            # Save pose
            np.save(str(pose_dir / f"{img_name}_pose.npy"), pose_mat)
            
            # Create visualization
            depth_viz = visualize_depth(depth_np)
            
            # Original image resized for visualization
            img1_resized = cv2.resize(img1, (args.img_width, args.img_height))
            
            # Create side-by-side visualization
            vis = np.hstack([img1_resized, depth_viz])
            cv2.imwrite(str(vis_dir / f"{img_name}_vis.png"), vis)
        
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            import traceback
            traceback.print_exc()

def visualize_depth(depth):
    """Create color visualization of depth map"""
    # Normalize depth for visualization
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)
    
    # Apply colormap
    colored_depth = (plt.cm.plasma(normalized_depth)[:, :, :3] * 255).astype(np.uint8)
    
    return colored_depth

def main():
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()
    
    # Load pretrained models
    try:
        disp_net, pose_net = load_pretrained_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process each organ subdirectory
    data_path = Path(args.data_path)
    for organ_dir in os.listdir(data_path):
        organ_path = data_path / organ_dir
        if os.path.isdir(organ_path):
            try:
                process_directory(organ_path, disp_net, pose_net)
            except Exception as e:
                print(f"Error processing directory {organ_dir}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()