from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import faiss
import numpy as np
from sklearn.cluster import KMeans
import shutil

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
            image_paths.append(img_path)  # Save full path for copying later
            print(f"Processed: {img_name}")

    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance for similarity
    index.add(embeddings)

    # Save the index and image paths for later use
    faiss.write_index(index, "image_index.faiss")
    np.save("image_paths.npy", np.array(image_paths))

    return index, image_paths, embeddings

# Perform clustering
def cluster_embeddings(embeddings, image_paths, n_clusters=5):
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

# Main workflow
folder_path = "./test for similarity thimira/set 2"
output_folder = "./output/set 2"

# Process images and cluster them
index, image_paths, embeddings = process_images(folder_path)
n_clusters = 5
clusters = cluster_embeddings(embeddings, image_paths, n_clusters=n_clusters)

# Save clustered images to folders
save_clustered_images(clusters, output_folder)
print(f"Clusters saved in {output_folder}")
