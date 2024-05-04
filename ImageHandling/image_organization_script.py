import os
import shutil

# Define your dataset directory and the path to the list.txt file
dataset_directory = 'path_to/oxford-iiit-pet/images/images'
instructions_file = 'path_to/oxford-iiit-pet/annotations/annotations/list.txt'

# Create directories for canines and felines if they don't exist
canine_dir = os.path.join(dataset_directory, 'canines')
feline_dir = os.path.join(dataset_directory, 'felines')

os.makedirs(canine_dir, exist_ok=True)
os.makedirs(feline_dir, exist_ok=True)

# Read the list.txt file and process each line
with open(instructions_file, 'r') as file:
    for line in file:
        # Skip comment lines
        if line.startswith('#'):
            continue

        # Extract the file name and species
        parts = line.split()

        print(parts)

        file_name, species = parts[0], parts[2]

        # Debug print to check what's being processed
        print(f"Processing {file_name} as {'Canine' if species == '2' else 'Feline'}")

        # Construct the full path to the image file
        file_path = os.path.join(dataset_directory, file_name + '.jpg')  # Adjust if different extensions are used

        # Move or copy the file to the corresponding directory
        if species == '1':  # Cat images
            shutil.copy(file_path, feline_dir)
        elif species == '2':  # Dog images
            shutil.copy(file_path, canine_dir)

print("Categorization complete!")
