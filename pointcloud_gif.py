import open3d as o3d
import numpy as np
import os
import tempfile
import imageio

def create_rotating_pointcloud_gif(ply_file_path, output_gif_path, fps=30, num_frames=90,
                                   point_size=2.0, background_color=[0.05, 0.05, 0.05],
                                   width=800, height=600):
    """
    Create a GIF of a single rotating point cloud with preserved colors
    
    Parameters:
    -----------
    ply_file_path : str
        Path to the PLY file
    output_gif_path : str
        Path to save the output GIF
    fps : int
        Frames per second for the GIF
    num_frames : int
        Total number of frames (determines smoothness)
    point_size : float
        Size of rendered points
    background_color : list
        RGB background color [0-1, 0-1, 0-1]
    width, height : int
        Resolution of the output GIF
    """
    print(f"Loading point cloud from {ply_file_path}...")
    pcd = o3d.io.read_point_cloud(ply_file_path)
    
    # Print information about the point cloud
    print(f"Point cloud has {len(pcd.points)} points")
    print(f"Point cloud has colors: {pcd.has_colors()}")
    
    # If no colors present in PLY, don't assign any (Open3D will use default)
    # This is important because assigning colors would override any vertex colors in the renderer
    
    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        vis.add_geometry(pcd)
        
        # Set render options
        render_option = vis.get_render_option()
        render_option.point_size = point_size
        render_option.background_color = np.array(background_color)
        
        # Important: Make sure point colors are enabled
        render_option.point_color_option = o3d.visualization.PointColorOption.Default
        
        # Get view control
        view_control = vis.get_view_control()
        
        # Setup initial camera view
        vis.reset_view_point(True)
        vis.poll_events()
        vis.update_renderer()
        
        # Adjust zoom to fit the model
        view_control.set_zoom(0.7)
        
        # Create frames
        frames = []
        print(f"Creating {num_frames} frames...")
        
        for frame in range(num_frames):
            # Rotate the camera around the point cloud
            rotation_angle = frame * 360.0 / num_frames
            
            # Update camera position based on rotation angle (circular path)
            radius = 2.0  # Distance from center
            # Use sine and cosine to create circular motion
            x = radius * np.sin(np.radians(rotation_angle))
            z = radius * np.cos(np.radians(rotation_angle))
            
            # Set the camera position
            view_control.set_front([x, 0, z])
            view_control.set_lookat([0, 0, 0])
            view_control.set_up([0, 1, 0])
            
            # Ensure scene is properly rendered
            vis.poll_events()
            vis.update_renderer()
            
            # Save the image
            frame_path = os.path.join(temp_dir, f"frame_{frame:04d}.png")
            vis.capture_screen_image(frame_path)
            frames.append(frame_path)
            
            # Show progress
            if frame % 10 == 0 or frame == num_frames - 1:
                print(f"Progress: {frame+1}/{num_frames} frames")
        
        vis.destroy_window()
        
        # Create the GIF
        print(f"Creating GIF with {len(frames)} frames...")
        with imageio.get_writer(output_gif_path, mode='I', fps=fps, loop=0) as writer:
            for frame_path in frames:
                image = imageio.imread(frame_path)
                writer.append_data(image)
        
        print(f"GIF saved to {output_gif_path}")

# Example usage
if __name__ == "__main__":
    create_rotating_pointcloud_gif(
        "/Users/aravjain/Documents/16824/projects/endoMonster/Final_data/Evaluations/Pointclouds/c3vd/000017_depth.ply",  # PLY file path
        "rotating_pointcloud.gif",      # Output GIF path
        fps=10,                         # Frames per second
        num_frames=100,                  # Total frames
        point_size=2.5                  # Point size
    )
