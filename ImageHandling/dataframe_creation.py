import os
import pandas as pd

# Set the base directories for canines and felines without using slashes in strings
base_directory = "ImageHandling"
image_directory = "images"
canines = "canines"
felines = "felines"


# Function to create a list of file paths and labels
def create_file_label_list(animal_directory, label):
    directory_path = os.path.join(image_directory, animal_directory)
    file_label_list = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(base_directory, directory_path, filename)
        file_label_list.append((file_path, label))
    return file_label_list


# Create lists for canines and felines
canine_list = create_file_label_list(canines, 'canine')
feline_list = create_file_label_list(felines, 'feline')

# Combine the lists
combined_list = canine_list + feline_list

# Create a DataFrame from the combined list and shuffle the rows
df = pd.DataFrame(combined_list, columns=['file_path', 'label']).sample(frac=1).reset_index(drop=True)

# Save the DataFrame to a CSV file (optional)
df.to_csv('image_labels.csv', index=False)

print(df.head())  # Print the first few entries to verify
