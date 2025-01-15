import os
import csv
from collections import defaultdict

# Step 1: Load data from the CSV file
def load_csv_data(csv_file):
    prefix_map = defaultdict(list)
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            prefix_map[row['prefix']].append(row['image_name'])
    return prefix_map

# Step 2: Verify folder contents match the prefix groupings
def verify_distribution(output_base_dir, csv_prefix_map):
    folder_prefix_map = defaultdict(list)

    # Read files from the distributed folders
    for folder_name in os.listdir(output_base_dir):
        folder_path = os.path.join(output_base_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                prefix = file_name[:50]  # Extract the prefix
                folder_prefix_map[prefix].append(file_name)

    # Compare CSV prefix map with folder contents
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
csv_file = 'folder_image_list_with_prefix.csv'
output_base_dir = 'generated/set 3'

# Load CSV data
csv_prefix_map = load_csv_data(csv_file)

# Verify distribution
verify_distribution(output_base_dir, csv_prefix_map)
