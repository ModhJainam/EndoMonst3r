import open3d as o3d
import os
import glob
import numpy as np

def view_pointcloud(ply_file_path):
    """
    Visualize a point cloud from a PLY file without coordinate arrows
    
    Parameters:
    -----------
    ply_file_path : str
        Path to the PLY file
    """
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(ply_file_path)
    
    # Print some basic information
    print(f"Point cloud contains {len(pcd.points)} points")
    print(f"Bounding box: {pcd.get_min_bound()} to {pcd.get_max_bound()}")
    
    # Visualize the point cloud without coordinate frame
    o3d.visualization.draw_geometries([pcd])

def view_multiple_pointclouds(directory, file_pattern="*.ply"):
    """
    View multiple point clouds from a directory
    
    Parameters:
    -----------
    directory : str
        Directory containing PLY files
    file_pattern : str, optional
        Pattern to match PLY files
    """
    # Get all PLY files in the directory
    ply_files = glob.glob(os.path.join(directory, file_pattern))
    
    print(f"Found {len(ply_files)} PLY files")
    
    for ply_file in ply_files:
        print(f"\nViewing: {os.path.basename(ply_file)}")
        view_pointcloud(ply_file)
        
        # Ask if user wants to continue to next file
        if len(ply_files) > 1:
            response = input("Press Enter to view next file, or 'q' to quit: ")
            if response.lower() == 'q':
                break

# Example usage
if __name__ == "__main__":
    # View a single file
    # view_pointcloud("path/to/your/pointcloud.ply")
    
    # Or view all PLY files in a directory
    view_multiple_pointclouds("/Users/aravjain/Documents/16824/projects/EndoSLAM/EndoSfMLearner/pointcloud_c3vd")