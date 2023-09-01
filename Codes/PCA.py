#%%
import pandas as pd
from sklearn.decomposition import PCA

# Load the dataset from the CSV file
csv_file_path = "C:/My_Project/output.csv"  # Replace with the path to your output CSV file
df = pd.read_csv(csv_file_path, header=None)

# Separate features (X) from labels (y)
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # Last column (labels)

# Perform PCA to determine the number of components to keep
pca = PCA()
pca.fit(X)

# Calculate the cumulative explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()

# Determine the number of components that capture a significant amount of variance
threshold_variance = 0.95  # You can adjust this threshold based on your preference
num_components = len(cumulative_variance_ratio[cumulative_variance_ratio <= threshold_variance])

print(f"Number of components to retain for {threshold_variance * 100:.2f}% explained variance: {num_components}")

# Initialize PCA with the determined number of components
pca = PCA(n_components=num_components)

# Perform PCA on the feature matrix X
X_pca = pca.fit_transform(X)

# Convert the PCA results back to a DataFrame
df_pca = pd.DataFrame(X_pca)

# Add the label column back to the DataFrame
df_pca['Label'] = y

# Save the PCA-transformed data to a new CSV file
pca_csv_file_path = "C:/My_Project/pca_output.csv"  # Replace with the desired PCA output CSV file path
df_pca.to_csv(pca_csv_file_path, index=False)
