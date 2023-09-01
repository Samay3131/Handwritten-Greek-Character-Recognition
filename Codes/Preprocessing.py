#%%
import os
import pandas as pd
from PIL import Image
from git import Repo
import shutil
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Define the mapping of characters to labels
label_mapping = {
    'ALPHA': 1, 'BETA': 2, 'GAMMA': 3, 'DELTA': 4, 'EPSILON': 5, 'ZETA': 6, 'HETA': 7, 'THETA': 8, 'IOTA': 9, 'KAPA': 10,
    'LAMDA': 11, 'MU': 12, 'NU': 13, 'KSI': 14, 'OMIKRON': 15, 'PII': 16, 'RO': 17, 'SIGMA': 18, 'TAU': 19, 'YPSILON': 20,
    'FI': 21, 'XI': 22, 'PSI': 23, 'OMEGA': 24
}

# Remove the existing directory if it exists
local_repo_path = "C:/My_Project/Greek_dataset/"
if os.path.exists(local_repo_path):
    # Remove files within .git directory manually
    git_dir = os.path.join(local_repo_path, ".git")
    for root, dirs, files in os.walk(git_dir, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.chmod(file_path, 0o777)  # Change file permission to be writable
            os.remove(file_path)

    # Remove the entire directory
    shutil.rmtree(local_repo_path)

# Clone the GitHub repository to the local directory
repo_url = "https://github.com/Samay3131/Greek_Dataset.git"  # Replace with your GitHub repository URL
Repo.clone_from(repo_url, local_repo_path)

git_dir = os.path.join(local_repo_path, ".git")
for root, dirs, files in os.walk(git_dir, topdown=False):
    for file in files:
        file_path = os.path.join(root, file)
        os.chmod(file_path, 0o777)  # Change file permission to be writable
        os.remove(file_path)
os.chmod(git_dir, 0o777)
shutil.rmtree(git_dir)
# Initialize an empty list to store the data

data = []

# Loop through each folder and process the images
folder_path = local_repo_path

for folder_name in os.listdir(folder_path):
    label = label_mapping.get(folder_name, 0)  # Set the label to 0 if folder_name not found in the mapping
    folder_full_path = os.path.join(folder_path, folder_name)
    image_files = [file for file in os.listdir(folder_full_path) if file.lower().endswith(('.png', '.jpg', '.bmp', '.gif'))]
    selected_images = random.sample(image_files, min(len(image_files), 100))


    for image_file in selected_images:
        image_path = os.path.join(folder_full_path, image_file)
        

        # Check if the current item is a file and skip if it's not an image file
        if not os.path.isfile(image_path) or not image_file.lower().endswith(('.png', '.jpg', '.bmp', '.gif')):
            continue

        image = Image.open(image_path).convert("L")  # Convert image to grayscale

        # Resize the image to 14x12 (196 pixels) and flatten it to get a 1D array
        resized_image = image.resize((14, 12))
        pixel_values = list(resized_image.getdata())

        # Add the label to the end of the pixel_values list
        pixel_values.append(label)

        data.append(pixel_values)

# Convert the data list to a pandas DataFrame
df = pd.DataFrame(data)


# Save the DataFrame to a CSV file
csv_file_path = "C:/My_Project/output.csv"  # Replace with the desired output CSV file path
if os.path.exists(csv_file_path):
    os.remove(csv_file_path)
    df.to_csv(csv_file_path, index=False, header=False, float_format='%.2f', mode='w')
else:
     df.to_csv(csv_file_path, index=False, header=False, float_format='%.2f', mode='w')
#################################################################################################
  
local_repo_path_2 = "C:/My_Project/Greek_dataset_1st/"

if os.path.exists(local_repo_path_2):
    # Remove files within .git directory manually
    git_dir = os.path.join(local_repo_path_2, ".git")
    for root, dirs, files in os.walk(git_dir, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.chmod(file_path, 0o777)  # Change file permission to be writable
            os.remove(file_path)
    shutil.rmtree(local_repo_path_2)

repo_url_2 = "https://github.com/Samay3131/Greek_dataSet_1st.git"  # Replace with your GitHub repository URL
Repo.clone_from(repo_url_2, local_repo_path_2)

git_dir = os.path.join(local_repo_path_2, ".git")
for root, dirs, files in os.walk(git_dir, topdown=False):
    for file in files:
        file_path = os.path.join(root, file)
        os.chmod(file_path, 0o777)  # Change file permission to be writable
        os.remove(file_path)
os.chmod(git_dir, 0o777)
shutil.rmtree(git_dir)