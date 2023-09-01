#%%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


data = []
labels = []
label_mapping = {}  # To map folder names to labels
data_folder = 'C:/My_Project/Greek_dataset'  

# Iterate through the subfolders, where each subfolder represents a character
for label, char_folder in enumerate(sorted(os.listdir(data_folder))):
    label += 1  # Start labels from 1
    char_folder_path = os.path.join(data_folder, char_folder)
    if os.path.isdir(char_folder_path):
        label_mapping[char_folder] = label  # Map folder name to label
        for image_file in os.listdir(char_folder_path):
            image_path = os.path.join(char_folder_path, image_file)
            image = Image.open(image_path)
            image = image.resize((64, 64))  # Resize image to 64x64
            data.append(np.array(image).flatten())  # Flatten the image
            labels.append(label)

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Create a DataFrame for EDA
df = pd.DataFrame(data=data, columns=[f"pixel_{i}" for i in range(data.shape[1])])
df['Label'] = labels

# Display basic statistics
print("Total images:", len(df))
print("Number of unique labels:", df['Label'].nunique())
print("Label mapping:", label_mapping)


# Display a few sample images
plt.figure(figsize=(12, 8))
for i in range(len(label_mapping)):
    label = i + 1
    sample_row = df[df['Label'] == label].iloc[0, :-1].values  # Get pixel values for the sample
    sample_image = sample_row.reshape(64, 64).astype(np.float32) / 255.0  # Adjust pixel values
    plt.subplot(4, 6, label)
    plt.imshow(sample_image, cmap='gray')
    plt.title(f"Label: {label} ({list(label_mapping.keys())[i]})")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Plot label distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Label', data=df)
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

num_components = 200  # Number of principal components
pca = PCA(n_components=num_components)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df.drop('Label', axis=1))  # Standardize data
df_pca = pd.DataFrame(pca.fit_transform(data_scaled), columns=[f'PC_{i}' for i in range(num_components)])
df_pca['Label'] = df['Label']

# Calculate basic statistics for each principal component
pca_stats = df_pca.describe()

# Print the statistics
print(pca_stats)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
plt.plot(cumulative_explained_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plots of pairs of principal components
sns.set(style="ticks")
sns.pairplot(df_pca.iloc[:, :4], diag_kind="kde")  # Plotting the first 4 principal components
plt.title("Scatter Plots of Principal Components")
plt.show()

# Calculate correlations between the first few principal components
num_pc_to_correlate = 10
pc_correlations = df_pca.iloc[:, :num_pc_to_correlate].corr()

# Print the correlations in a table format
print("Correlation Matrix for the First", num_pc_to_correlate, "Principal Components:")
print(pc_correlations)

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlations between the first 10 principal components
pc_correlations = df_pca.iloc[:, :10].corr()

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pc_correlations, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix of the First 10 Principal Components")
plt.show()

# %%
