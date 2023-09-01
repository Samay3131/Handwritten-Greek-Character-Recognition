#%%
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_val_test_split(X, Y, val_size, test_size, random_state):
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=val_size + test_size, random_state=random_state)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=test_size / (val_size + test_size), random_state=random_state)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
# Function to sort an array
def sort_distance(e):
    for i in range(len(e)):
        for j in range(i + 1, len(e)):
            swap = 0
            if e[i] > e[j]:
                swap = e[i]
                e[i] = e[j]
                e[j] = swap
    return e

# Function to find the most common nearest neighbor label
def split_neighbours(pred):
    y_pred = []
    for i in range(len(pred)):
        temp = []
        temp = list(pred[i])
        c = []
        for i in temp:
            c.append(temp.count(i))
            index_c = c.index(max(c))
        y_pred.append(temp[index_c])
    return y_pred

# Function to calculate accuracy and error rate
def accuracy_calculate(y_pred, y_test):
    acc = y_pred == y_test
    ct = 0
    cf = 0
    for i in acc:
        if i == True:
            ct = ct + 1
        if i == False:
            cf = cf + 1
            
    accuracy = 1 - (cf / (cf + ct)) 
    error_rate = 1 - accuracy
    return error_rate, accuracy

# Function to predict labels using k-nearest neighbors
import math as m

def predict(K, x_test, X_train, Y_train):
    pred = []
    for i in range(len(x_test)):
        index = []
        neighbour = []
        distance = []
        dist = []
        for j in range(len(X_train)):
            distance.append(m.sqrt(sum((x_test[i] - X_train[j])**2)))
        dist.extend(distance)
        eucl = sort_distance(dist)
        for i in range(len(distance)):
            for j in range(len(eucl)):
                if eucl[i] == distance[j]:
                    index.append(j)
        neighbour.append(index[:K])
        for i in neighbour:
            pred.append(Y_train[i])
    pred = np.asarray(pred)
    y_pred = split_neighbours(pred)
    y_pred = np.asarray(y_pred)
    return y_pred

def find_best_K(X_train, y_train, X_val, y_val, max_k,title):
    best_K = None
    highest_val_accuracy = 0.0
    min_train_accuracy_difference = 0.1  # Minimum difference between train and validation accuracy

    train_accuracies = []  # List to store train accuracies for all K values
    val_accuracies = []    # List to store validation accuracies for all K values

    for K in range(1, max_k + 1):
        # Calculate training and validation accuracy for the current K
        y_train_pred = predict(K, X_train, X_train, y_train)
        train_error_rate, train_accuracy = accuracy_calculate(y_train_pred, y_train)
        y_val_pred = predict(K, X_val, X_train, y_train)
        val_error_rate, val_accuracy = accuracy_calculate(y_val_pred, y_val)
        
        # Check if the validation accuracy is higher and the train accuracy is not 100%
        if val_accuracy > highest_val_accuracy and train_accuracy < 1.0 and train_accuracy > 1.0 - min_train_accuracy_difference:
            best_K = K
            highest_val_accuracy = val_accuracy
        
        # Append accuracies to the lists
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Print the current K and its corresponding train and validation accuracies
        print(f"K = {K}: Train Accuracy = {train_accuracy:.4f}, Validation Accuracy = {val_accuracy:.4f}")

    # Plot train and validation accuracies
    plt.plot(range(1, max_k + 1), train_accuracies, marker='o', label='Train Accuracy')
    plt.plot(range(1, max_k + 1), val_accuracies, marker='o', label='Validation Accuracy')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.savefig(f'C:/My_Project/{title}.png')
    plt.show(block=False)

    # Print the best K value
    print(f"\nBest K value based on modified criteria: {best_K}")

    return best_K

def plot_error_rate_vs_k(X_train, y_train, X_test, y_test, max_k, title):
    k_values = range(1, max_k + 1)
    error_rates = []
    for k in k_values:
        y_pred = predict(k, X_test, X_train, y_train)
        error_rate, _ = accuracy_calculate(y_pred, y_test)
        error_rates.append(error_rate)
    plt.plot(k_values, error_rates, marker='o')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.title(title) 
    plt.savefig(f'C:/My_Project/{title}.png')
    plt.show(block=False) 

def plot_accuracy_vs_k(X_train, y_train, X_test, y_test, max_k, title):
    k_values = range(1, max_k + 1)
    accuracies = []
    for k in k_values:
        y_pred = predict(k, X_test, X_train, y_train)
        _, accuracy = accuracy_calculate(y_pred, y_test)
        accuracies.append(accuracy)
    plt.plot(k_values, accuracies, marker='o', color='orange')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.savefig(f'C:/My_Project/{title}.png')
    plt.show(block=False) 

def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title) 
    plt.savefig(f'C:/My_Project/{title}.png')
    plt.show(block=False) 

# Load the Greek classification dataset
train_data = pd.read_csv('C:/My_Project/Greek_dataset_1st/train.csv')
test_data = pd.read_csv('C:/My_Project/Greek_dataset_1st/test.csv')

X_train_full = np.asarray(train_data.iloc[:, :-1])
y_train_full = np.asarray(train_data.iloc[:, -1])
X_test = np.asarray(test_data.iloc[:, :-1])
y_test = np.asarray(test_data.iloc[:, -1])

# Set the maximum value for K to 20
max_k = 20
from sklearn.model_selection import train_test_split
# Find the best K value for Greek Classification Dataset
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=3105)
best_K_greek = find_best_K(X_train, y_train, X_val, y_val, max_k,"Train and Validation accuracy Vs K - Greek Classification")

# Use the best K value for Greek Classification Dataset to calculate and print the final test accuracy
y_pred_best_greek = predict(best_K_greek, X_test, X_train_full, y_train_full)
final_error_rate_greek, final_accuracy_greek = accuracy_calculate(y_pred_best_greek, y_test)
print(f"\nFinal Test Accuracy with Best K ({best_K_greek}) for Greek Classification Dataset: {final_accuracy_greek:.4f}")

# Plot error rate and accuracy vs. K for the Greek Classification Dataset
plot_error_rate_vs_k(X_train_full, y_train_full, X_test, y_test, max_k, "Error Rate vs. K - Greek Classification Dataset")
# Plot accuracy vs. K for the Greek Classification Dataset
plot_accuracy_vs_k(X_train_full, y_train_full, X_test, y_test, max_k, "Accuracy vs. K - Greek Classification Dataset")

# Plot confusion matrix for Greek Classification Dataset
plot_confusion_matrix(y_test, y_pred_best_greek, np.unique(y_train_full), "KNN Confusion matrix for Greek Classification Dataset")

############################################################################
# # Load PCA output data
pca_output_data = pd.read_csv('C:/My_Project/pca_output.csv', header=None)
X_pca = np.asarray(pca_output_data.iloc[:, :-1])
y_pca = np.asarray(pca_output_data.iloc[:, -1])

# # Split the PCA data into training, validation, and test sets
X_train_pca, X_val_pca, x_test_pca, y_train_pca, y_val_pca, y_test_pca = train_val_test_split(X_pca, y_pca, val_size=0.1, test_size=0.1, random_state=3105)

# Find the best K value for PCA output data
best_K_pca = find_best_K(X_train_pca, y_train_pca, X_val_pca, y_val_pca, max_k,"Train and Validation accuracy Vs K - GCDB Dataset")

# Use the best K value for PCA output data to calculate and print the final test accuracy
y_pred_best_pca = predict(best_K_pca, x_test_pca, X_train_pca, y_train_pca)
final_error_rate_pca, final_accuracy_pca = accuracy_calculate(y_pred_best_pca, y_test_pca)
print(f"\nFinal Test Accuracy with Best K ({best_K_pca}) for PCA Output Data: {final_accuracy_pca:.4f}")

plot_error_rate_vs_k(X_train_pca, y_train_pca, x_test_pca, y_test_pca, max_k, "Error Rate vs. K - GCDB Dataset")
plot_accuracy_vs_k(X_train_pca, y_train_pca, x_test_pca, y_test_pca, max_k, "Accuracy vs. K - GCDB Dataset")

# # Plot confusion matrix for PCA output data
plot_confusion_matrix(y_test_pca, y_pred_best_pca, np.unique(y_pca), "KNN Confusion matrix for GCDB Dataseta")



# # Finally, the code execution ends here

