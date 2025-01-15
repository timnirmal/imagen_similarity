import tkinter as tk
from tkinter import filedialog, messagebox
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import faiss
import numpy as np
from sklearn.cluster import KMeans
import shutil
from collections import defaultdict
from itertools import cycle
import csv

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to extract image embeddings
def extract_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy().flatten()

# Process all images in a folder and store embeddings in FAISS
def process_images(folder_path):
    image_paths = []
    embeddings = []

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            embedding = extract_image_embedding(img_path)
            embeddings.append(embedding)
            image_paths.append(img_path)
            print(f"Processed: {img_name}")

    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance for similarity
    index.add(embeddings)

    return index, image_paths, embeddings

# Perform clustering
def cluster_embeddings(embeddings, image_paths, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(image_paths[idx])
    return clusters

# Create output folders and copy images
def save_clustered_images(clusters, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for cluster_id, images in clusters.items():
        cluster_folder = os.path.join(output_folder, str(cluster_id + 1))
        os.makedirs(cluster_folder, exist_ok=True)
        for image_path in images:
            shutil.copy(image_path, cluster_folder)
            print(f"Copied {image_path} to {cluster_folder}")

# Create CSV with prefixes
def create_csv_with_prefix(base_dir, output_csv, prefix_length=50):
    data = []
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                prefix = image_name[:prefix_length]
                data.append([folder_name, image_name, prefix])

    # Write to CSV
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['folder_name', 'image_name', 'prefix'])
        writer.writerows(data)

    print(f"Data with prefixes saved to {output_csv}")
    return data

# Distribute images into existing numbered folders
def distribute_images(base_dir, output_base_dir, data, existing_folders):
    prefix_map = defaultdict(list)
    for folder_name, image_name, prefix in data:
        image_path = os.path.join(base_dir, folder_name, image_name)
        prefix_map[prefix].append(image_path)

    folder_cycle = cycle(existing_folders)
    folder_assignments = {}

    for prefix, files in prefix_map.items():
        if prefix not in folder_assignments:
            folder_assignments[prefix] = next(folder_cycle)

        target_folder = os.path.join(output_base_dir, str(folder_assignments[prefix]))
        os.makedirs(target_folder, exist_ok=True)
        for file_path in files:
            target_path = os.path.join(target_folder, os.path.basename(file_path))
            shutil.move(file_path, target_path)
            print(f"Moved {file_path} to {target_folder}")

def load_csv_data(csv_file):
    prefix_map = defaultdict(list)
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            prefix_map[row['prefix']].append(row['image_name'])
    return prefix_map


# Verify folder contents
def verify_distribution(output_base_dir, csv_prefix_map):
    folder_prefix_map = defaultdict(list)

    for folder_name in os.listdir(output_base_dir):
        folder_path = os.path.join(output_base_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                prefix = file_name[:50]
                folder_prefix_map[prefix].append(file_name)

    mismatches = []
    for prefix, csv_files in csv_prefix_map.items():
        folder_files = folder_prefix_map.get(prefix, [])
        if set(csv_files) != set(folder_files):
            mismatches.append({
                'prefix': prefix,
                'csv_files': csv_files,
                'folder_files': folder_files
            })

    if mismatches:
        print("Verification failed! Mismatches found:")
        for mismatch in mismatches:
            print(f"\nPrefix: {mismatch['prefix']}")
            print(f"Expected (from CSV): {mismatch['csv_files']}")
            print(f"Found in folder: {mismatch['folder_files']}")
    else:
        print("Verification successful! All files are correctly distributed.")

# Main workflow
def main(folder_path, output_folder, n_clusters):
    # Step 1: Process images
    index, image_paths, embeddings = process_images(folder_path)

    # Step 2: Cluster embeddings
    clusters = cluster_embeddings(embeddings, image_paths, n_clusters)

    # Step 3: Save clustered images
    save_clustered_images(clusters, output_folder)

    # Step 4: Create CSV with prefixes
    csv_file = 'folder_image_list_with_prefix.csv'
    create_csv_with_prefix(output_folder, csv_file)

    # Step 5: Distribute images
    distribute_images(output_folder, f"generated/{os.path.basename(output_folder)}", create_csv_with_prefix(output_folder, csv_file), list(range(1, n_clusters + 1)))

    # Step 6: Verify distribution
    csv_prefix_map = load_csv_data(csv_file)
    verify_distribution(f"generated/{os.path.basename(output_folder)}", csv_prefix_map)

# GUI Application
def run_gui():
    def start_processing():
        folder_path = input_folder_var.get()
        output_folder = "./output"
        try:
            n_clusters = int(cluster_entry.get())
        except ValueError:
            messagebox.showwarning("Warning", "Number of clusters must be an integer.")
            return

        if folder_path and n_clusters > 0:
            try:
                main(folder_path, output_folder, n_clusters)
                messagebox.showinfo("Success", f"Processing complete! Clusters saved in {output_folder}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
        else:
            messagebox.showwarning("Warning", "Please provide valid input folder and number of clusters.")

    def select_input_folder():
        folder = filedialog.askdirectory(title="Select Input Folder")
        input_folder_var.set(folder)

    root = tk.Tk()
    root.title("Image Clustering Tool")
    root.geometry("500x300")

    tk.Label(root, text="Image Clustering Tool", font=("Helvetica", 16, "bold")).pack(pady=10)

    input_folder_var = tk.StringVar()

    tk.Button(root, text="Select Input Folder", command=select_input_folder).pack(pady=5)
    tk.Entry(root, textvariable=input_folder_var, width=50).pack(pady=5)

    tk.Label(root, text="Number of Clusters:").pack(pady=5)
    cluster_entry = tk.Entry(root)
    cluster_entry.pack(pady=5)

    tk.Button(root, text="Start Processing", command=start_processing, bg="green", fg="white", font=("Helvetica", 12, "bold")).pack(pady=20)

    root.mainloop()

# Example usage
if __name__ == "__main__":
    run_gui()
