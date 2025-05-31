import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
import os
import matplotlib.pyplot as plt
import open3d as o3d  # For 3D point cloud visualization

def load_depth_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    model.eval()
    return model

def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")  
        img = img.resize((384, 384))  
        img = np.array(img) / 255.0  
        return img
    except PermissionError:
        print(f"Permission denied for file: {image_path}")
        return None

def estimate_depth(model, img):
    if img is None:
        return None
    
    input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  
    with torch.no_grad():
        depth_map = model(input_tensor)
    return depth_map.squeeze().numpy()

def normalize_depth_map(depth_map):
    """Normalize depth map for visualization."""
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    normalized_depth = (depth_map - depth_min) / (depth_max - depth_min) * 255.0
    return normalized_depth.astype(np.uint8)

def save_depth_image(depth_map, output_path):
    """Save depth map as an image."""
    normalized_depth = normalize_depth_map(depth_map)
    depth_image = Image.fromarray(normalized_depth)
    depth_image.save(output_path)
    print(f"Depth map saved to {output_path}")

def generate_point_cloud(depth_map):
    if depth_map is None:
        return None
    
    h, w = depth_map.shape
    points = []
    
    for y in range(h):
        for x in range(w):
            z = depth_map[y, x]
            points.append([x, y, z])
    
    return np.array(points)

def visualize_point_cloud_open3d(point_cloud):
    """Visualize point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud")

def visualize_point_cloud_matplotlib(point_cloud):
    """Visualize point cloud using Matplotlib (slower but simpler)."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c=point_cloud[:, 2], cmap='viridis')
    ax.set_title("3D Point Cloud (Matplotlib)")
    plt.show()

def create_mesh(point_cloud):
    if point_cloud is None:
        return None
    
    mesh = trimesh.Trimesh(vertices=point_cloud)
    return mesh

def export_mesh(mesh, output_path):
    if mesh is None:
        print("Mesh is None. Cannot export.")
        return
    
    mesh.export(output_path)

def main(image_path, output_model_path, output_depth_image_path):
    print("Loading depth estimation model...")
    model = load_depth_model()
    
    print("Loading and preprocessing image...")
    img = preprocess_image(image_path)
    
    if img is None:
        print("Failed to load the image. Exiting.")
        return
    
    print("Estimating depth...")
    depth_map = estimate_depth(model, img)
    
    if depth_map is None:
        print("Failed to estimate depth. Exiting.")
        return
    
    print("Saving depth map as image...")
    save_depth_image(depth_map, output_depth_image_path)
    
    print("Generating point cloud...")
    point_cloud = generate_point_cloud(depth_map)
    
    if point_cloud is None:
        print("Failed to generate point cloud. Exiting.")
        return
    
    print("Visualizing point cloud (Open3D)...")
    visualize_point_cloud_open3d(point_cloud)
    
    print("Visualizing point cloud (Matplotlib)...")
    visualize_point_cloud_matplotlib(point_cloud)
    
    print("Creating mesh...")
    mesh = create_mesh(point_cloud)
    
    if mesh is None:
        print("Failed to create mesh. Exiting.")
        return
    
    print("Exporting 3D model...")
    export_mesh(mesh, output_model_path)
    
    print(f"3D model exported to {output_model_path}")
    print(f"Depth map saved to {output_depth_image_path}")

if __name__ == "__main__":
    input_image_path = r'E:\Major Project 2024\2D to 3D\image input\i3.jpg' 
    output_model_path = r'E:\Major Project 2024\2D to 3D\image output\image.obj'       
    output_depth_image_path = r'E:\Major Project 2024\2D to 3D\image output\depth_map.png'
    
    main(input_image_path, output_model_path, output_depth_image_path)