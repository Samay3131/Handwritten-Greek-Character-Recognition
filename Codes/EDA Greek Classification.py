#%%

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


train_folder = 'C:/My_Project/Greek_dataset_1st/train_letters_images/'
test_folder = 'C:/My_Project/Greek_dataset_1st/test_letters_images/'

# Load a few images from the train and test folders
train_images = os.listdir(train_folder)
test_images = os.listdir(test_folder)

print(f"Number of train images: {len(train_images)}")
print(f"Number of test images: {len(test_images)}")



# Common size for resizing
target_size = (100, 100)  # Adjust the size as needed

# Initialize list to store image data
images = []

# Loop through the image files in the train folder
for filename in os.listdir(train_folder):
    if filename.endswith('.jpg'):  # Assuming images are in .jpg format
        image_path = os.path.join(train_folder, filename)
        
        # Load and resize the image
        img = Image.open(image_path).resize(target_size)
        img_array = np.array(img)
        
        images.append(img_array)



# Convert images to a numpy array
images_array = np.array(images)

# Flatten the images
flattened_images = images_array.reshape(images_array.shape[0], -1)

# Perform k-means clustering
num_clusters = 24  
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(flattened_images)

# Visualize images from the first 5 clusters
sample_size = 5
for cluster in range(min(5, num_clusters)):
    cluster_indices = [i for i, c in enumerate(clusters) if c == cluster]
    
    if len(cluster_indices) >= sample_size:
        sample_indices = np.random.choice(cluster_indices, size=sample_size, replace=False)
    else:
        sample_indices = cluster_indices
    
    plt.figure(figsize=(10, 5))
    for i, index in enumerate(sample_indices):
        img = images_array[index]
        plt.subplot(1, sample_size, i + 1)
        plt.imshow(img, cmap='gray')  
        plt.title(f"Cluster {cluster + 1}")
        plt.axis('off')
    plt.show()


from sklearn.manifold import TSNE
import seaborn as sns

# Perform t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(flattened_images)

# Create a scatter plot to visualize clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=clusters, palette='tab20', legend='full')
plt.title('t-SNE Visualization of Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(loc='upper right')
plt.show()



from scipy.stats import norm

# Pixel Distribution Analysis
mean_pixel_values = np.mean(flattened_images, axis=1)
std_pixel_values = np.std(flattened_images, axis=1)
variance_pixel_values = np.var(flattened_images, axis=1)
median_pixel_values = np.median(flattened_images, axis=1)
print("Medain Pixel Values")
print(mean_pixel_values)
print("Standard deviation of pixel Values")
print(std_pixel_values)
print("Varience of pixel values")
print(variance_pixel_values)
print("Medain of Pixel Values")
print(median_pixel_values)
# Create line plots to visualize pixel distribution statistics with normal-like distribution curves
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(mean_pixel_values, bins=20, density=True, color='blue', alpha=0.7)
x_range = np.linspace(np.min(mean_pixel_values), np.max(mean_pixel_values), 100)
plt.plot(x_range, norm.pdf(x_range, np.mean(mean_pixel_values), np.std(mean_pixel_values)), color='red', linewidth=2)
plt.title('Mean Pixel Values')

plt.subplot(2, 2, 2)
plt.hist(std_pixel_values, bins=20, density=True, color='green', alpha=0.7)
x_range = np.linspace(np.min(std_pixel_values), np.max(std_pixel_values), 100)
plt.plot(x_range, norm.pdf(x_range, np.mean(std_pixel_values), np.std(std_pixel_values)), color='red', linewidth=2)
plt.title('Standard Deviation of Pixel Values')

plt.subplot(2, 2, 3)
plt.hist(variance_pixel_values, bins=20, density=True, color='purple', alpha=0.7)
x_range = np.linspace(np.min(variance_pixel_values), np.max(variance_pixel_values), 100)
plt.plot(x_range, norm.pdf(x_range, np.mean(variance_pixel_values), np.std(variance_pixel_values)), color='red', linewidth=2)
plt.title('Variance of Pixel Values')

plt.subplot(2, 2, 4)
plt.hist(median_pixel_values, bins=20, density=True, color='orange', alpha=0.7)
x_range = np.linspace(np.min(median_pixel_values), np.max(median_pixel_values), 100)
plt.plot(x_range, norm.pdf(x_range, np.mean(median_pixel_values), np.std(median_pixel_values)), color='red', linewidth=2)
plt.title('Median Pixel Values')

plt.tight_layout()
plt.show()

# %%
