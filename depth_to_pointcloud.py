import numpy as np
import open3d as o3d
import os
import glob
from tqdm import tqdm  # For progress tracking

def depth_to_pointcloud(depth_map_path, fx, fy, cx, cy, save_path=None):
    """
    Convert a depth map (.npy file) to a point cloud
    
    Parameters:
    -----------
    depth_map_path : str
        Path to the .npy depth map file
    fx, fy : float
        Focal lengths in x and y directions
    cx, cy : float
        Principal point coordinates
    save_path : str, optional
        Path to save the point cloud (in .ply format)
        
    Returns:
    --------
    pcd : open3d.geometry.PointCloud
        Point cloud object
    """
    # Load depth map
    depth = np.load(depth_map_path)
    
    # Get image dimensions
    height, width = depth.shape
    
    # Create pixel coordinates grid
    v, u = np.mgrid[0:height, 0:width]
    
    # Convert from pixel coordinates to camera coordinates
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack coordinates
    points = np.stack((x, y, z), axis=-1)
    
    # Reshape to list of points
    points = points.reshape(-1, 3)
    
    # Remove points with zero or invalid depth
    mask = (z.flatten() > 0) & (z.flatten() < np.inf)
    points = points[mask]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals (optional)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Save if requested
    if save_path:
        o3d.io.write_point_cloud(save_path, pcd)
    
    return pcd

def process_multiple_depth_maps(input_dir, output_dir, fx, fy, cx, cy, file_pattern="*.npy"):
    """
    Process multiple depth maps from a directory
    
    Parameters:
    -----------
    input_dir : str
        Directory containing .npy depth map files
    output_dir : str
        Directory to save point cloud files
    fx, fy : float
        Focal lengths in x and y directions
    cx, cy : float
        Principal point coordinates
    file_pattern : str, optional
        Pattern to match depth map files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .npy files in the input directory
    depth_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    print(f"Found {len(depth_files)} depth map files to process")
    
    # Process each file
    for i, depth_file in enumerate(tqdm(depth_files)):
        # Get filename without extension
        filename = os.path.splitext(os.path.basename(depth_file))[0]
        
        # Define output path
        output_path = os.path.join(output_dir, f"{filename}.ply")
        
        # Process the file
        try:
            pcd = depth_to_pointcloud(depth_file, fx, fy, cx, cy, output_path)
            # Optional: Do something with the point cloud
        except Exception as e:
            print(f"Error processing {depth_file}: {e}")

# Example usage
if __name__ == "__main__":
    # Replace with your camera parameters
    #For EndoSlam
    fx, fy = 178.5604, 156.0418   # Focal length
    cx, cy = 181.8043, 155.7529  # Principal point

    #For C3VD
    fx, fy = 769.2436, 769.2436   # Focal length
    cx, cy = 678.5448, 542.9759  # Principal point
    
    # Directory with depth maps
    input_dir = "/Users/aravjain/Documents/16824/projects/endoMonster/Final_data/Output/c3vd_output/cecum_t1_a/depth"
    
    # Directory to save point clouds
    output_dir = "/Users/aravjain/Documents/16824/projects/endoMonster/Final_data/Evaluations/Pointclouds/c3vd"
    
    # Process all depth maps
    process_multiple_depth_maps(input_dir, output_dir, fx, fy, cx, cy)
    
    # Optional: Visualize one of the point clouds
    # pcd = o3d.io.read_point_cloud(os.path.join(output_dir, "example.ply"))
    # o3d.visualization.draw_geometries([pcd])
    
