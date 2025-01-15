import os
import csv
import shutil
from collections import defaultdict
from itertools import cycle  # Import cycle for circular iteration

# Step 1: Collect folder names, image names, and extract prefixes by character length
def create_csv_with_prefix(base_dir, output_csv, prefix_length=50):
    data = []
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                prefix = image_name[:prefix_length]  # Extract prefix by character length
                data.append([folder_name, image_name, prefix])

    # Write to CSV
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['folder_name', 'image_name', 'prefix'])
        writer.writerows(data)

    print(f"Data with prefixes saved to {output_csv}")
    return data

# Step 2: Distribute images into existing numbered folders
def distribute_images(base_dir, output_base_dir, data, existing_folders):
    # Group images by their prefix
    prefix_map = defaultdict(list)
    for folder_name, image_name, prefix in data:
        image_path = os.path.join(base_dir, folder_name, image_name)
        prefix_map[prefix].append(image_path)

    # Use cycle to iterate through folders circularly
    folder_cycle = cycle(existing_folders)  # Infinite cycling through folders
    folder_assignments = {}  # Keep track of which folder each prefix is assigned to

    for prefix, files in prefix_map.items():
        # Assign the prefix to a folder
        if prefix not in folder_assignments:
            folder_assignments[prefix] = next(folder_cycle)

        target_folder = os.path.join(output_base_dir, str(folder_assignments[prefix]))

        # Move files to the target folder
        os.makedirs(target_folder, exist_ok=True)
        for file_path in files:
            target_path = os.path.join(target_folder, os.path.basename(file_path))
            shutil.move(file_path, target_path)
            print(f"Moved {file_path} to {target_folder}")

    print(f"Images distributed into existing folders: {existing_folders}")

# Main workflow
base_dir = 'output/set 3'
output_csv = 'folder_image_list_with_prefix.csv'
output_base_dir = 'generated/set 3'
prefix_length = 50  # Set prefix length
existing_folders = [1, 2, 3, 4, 5]  # Existing numbered folders

# Step 1: Create CSV with prefixes
data = create_csv_with_prefix(base_dir, output_csv, prefix_length)

# Step 2: Distribute images into numbered folders
distribute_images(base_dir, output_base_dir, data, existing_folders)
