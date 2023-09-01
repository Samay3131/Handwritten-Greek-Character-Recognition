#%%

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd 
import random
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model 
from PIL import Image

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Check and configure GPU usage
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    print("No GPUs available.")
else:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define data paths
data_dir = 'C:/My_Project/Greek_dataset'  # Replace with the path to your main dataset folder

# Create a list to store image paths and corresponding labels
image_paths = []
labels = []

# Loop through subfolders to gather image paths and labels
for label, char_folder in enumerate(sorted(os.listdir(data_dir))):
    char_folder_path = os.path.join(data_dir, char_folder)
    for image_name in os.listdir(char_folder_path):
        image_path = os.path.join(char_folder_path, image_name)
        image_paths.append(image_path)
        labels.append(label)

# Split the data into train, validation, and test sets
train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=seed)
train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.2, random_state=seed)

# Convert labels to string representation
train_labels = [str(label) for label in train_labels]
val_labels = [str(label) for label in val_labels]
test_labels = [str(label) for label in test_labels]

# Create an ImageDataGenerator for data normalization and augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    # brightness_range=[0.5, 1.5],
    rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # horizontal_flip=True
)

# Load and organize the data using the ImageDataGenerator
batch_size = 32
train_generator = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': train_paths, 'class': train_labels}),
    x_col='filename',
    y_col='class',
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical',  # Use categorical labels for multiclass classification
    shuffle=True
)
val_generator = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': val_paths, 'class': val_labels}),
    x_col='filename',
    y_col='class',
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
test_generator = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': test_paths, 'class': test_labels}),
    x_col='filename',
    y_col='class',
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define the CNN model with increased complexity
num_classes = len(np.unique(labels))
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Use num_classes here
])
def plot_training_curves(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy for GCDB Dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('C:/My_Project/accuracy_curves_CNN.png')
    plt.show(block=False) 
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss for GCDB Dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('C:/My_Project/loss_curves_CNN.png')
    plt.show(block=False) 
    
# Compile the model

    
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
plot_path = 'C:/My_Project/model_architecture_cnn2.png'
plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
image = Image.open(plot_path)
horizontal_image = image.rotate(90, expand=True)
horizontal_image.save(plot_path)
horizontal_image.show()
# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)



# Train the model
num_epochs = 10
history  = model.fit(train_generator, epochs=num_epochs, validation_data=val_generator, callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
plot_training_curves(history)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

# Predict class probabilities for the test set
test_predictions = model.predict(test_generator)
predicted_classes = np.argmax(test_predictions, axis=1)

# Create the confusion matrix
confusion = confusion_matrix(test_generator.classes, predicted_classes)

# Create a DataFrame for better visualization
confusion_df = pd.DataFrame(confusion, index=test_generator.class_indices.keys(), columns=test_generator.class_indices.keys())

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_df, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for CNN For GCDB Dataset')
plt.savefig('C:/My_Project/confusion_matrix_CNN2.png')
plt.show(block=False) 

# %%
