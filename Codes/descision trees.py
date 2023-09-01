# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Ignore UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)




class DecisionTree:
    def __init__(self, max_depth=None): 
        self.max_depth = max_depth
        self.tree = {}
    

    def gini_index(self, y):
        classes = np.unique(y)
        n = len(y)
        gini = 1.0

        for c in classes:
            p = np.count_nonzero(y == c) / n
            gini -= p ** 2

        return gini

    def best_split(self, X, y):
        n_features = X.shape[1]
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold
                gini = (self.gini_index(y[left_indices]) * np.sum(left_indices) +
                        self.gini_index(y[right_indices]) * np.sum(right_indices)) / len(y)

                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def build_tree(self, X, y, depth=0):
        if self.max_depth is not None and depth == self.max_depth or len(np.unique(y)) == 1:
            return {'label': np.argmax(np.bincount(y))}

        feature_index, threshold = self.best_split(X, y)
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        left = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self.build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature_index': feature_index, 'threshold': threshold, 'left': left, 'right': right}

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_sample(self, x, tree):
        while 'feature_index' in tree:
                                

            if x[tree['feature_index']] <= tree['threshold']:
                tree = tree['left']
            else:
                tree = tree['right']
        return tree['label']

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.predict_sample(x, self.tree))
        return predictions
    
    def accuracy(self, y_pred, y_test):
        acc = y_pred == y_test
        ct = 0
        cf = 0
        for i in acc:
            if i == True:
                ct += 1  # calculating the count of True predicted values
            if i == False:
                cf += 1  # calculating the count of False predicted values

        accuracy = ct / (cf + ct)
        error_rate = 1 - accuracy
        return accuracy, error_rate

def data_split(data):
    X = np.asarray(data.iloc[:, :-1]).astype(float)
    y = np.asarray(data.iloc[:, -1]).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state= 3105)
    
    return X_train, X_test, y_train, y_test



    
    

def class_wise_accuracy(y_true, y_pred, y_pred_rf):
    classes = np.unique(y_true)
    accuracies_dt = []
    accuracies_rf = []

    for cls in classes:
        mask = y_true == cls
        y_true_class = y_true[mask]
        y_pred_class = np.array(y_pred)[mask]
        y_pred_rf_class = y_pred_rf[mask]

        accuracy_dt = accuracy_score(y_true_class, y_pred_class)
        accuracies_dt.append(accuracy_dt)

        accuracy_rf = accuracy_score(y_true_class, y_pred_rf_class)
        accuracies_rf.append(accuracy_rf)

    return classes, accuracies_dt, accuracies_rf





def plot_class_wise_accuracy(classes, accuracies_dt, accuracies_rf, dataset_name):
    plt.figure(figsize=(10, 6))
    plt.bar(classes - 0.2, accuracies_dt, width=0.4, label='Decision Tree', align='center')
    plt.bar(classes + 0.2, accuracies_rf, width=0.4, label='Random Forest', align='center')
    plt.xticks(classes)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title(f'Class-wise Accuracy Comparison for {dataset_name}')
    plt.legend()
    plt.savefig(f'C:/My_Project/class_wise_accuracy_{dataset_name}.png')
    plt.show(block=False)   

def plot_combined_confusion_matrix(y_true, y_pred_dt, y_pred_rf, dataset_name):
    num_labels = 24  
    cm_dt = confusion_matrix(y_true, y_pred_dt, labels=np.arange(1, num_labels + 1))
    cm_rf = confusion_matrix(y_true, y_pred_rf, labels=np.arange(1, num_labels + 1))
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Decision Tree Confusion Matrix")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    plt.tight_layout()
    save_path = f"C:/My_Project/combined_confusion_matrix_{dataset_name}.png"
    plt.savefig(save_path)
    plt.show()





def model_with_cross_validation(X_train, y_train, X_test, y_test, max_depth_range, dataset_name):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dt_train_accuracies = []
    dt_test_accuracies = []
    rf_train_accuracies = []
    rf_test_accuracies = []
    
    dt_cv_scores = []  # New list to store cross-validation scores for Decision Tree
    rf_cv_scores = []  # New list to store cross-validation scores for Random Forest
    best_dt_params = None  # Initialize to None
    best_rf_params = None  # Initialize to None

    for max_depth in max_depth_range:
        dt_train_accuracy_sum = 0
        dt_test_accuracy_sum = 0
        rf_train_accuracy_sum = 0
        rf_test_accuracy_sum = 0
        
        dt_cv_scores_for_depth = []  # List to store CV scores for Decision Tree for this depth
        rf_cv_scores_for_depth = []  # List to store CV scores for Random Forest for this depth

        for train_index, test_index in kf.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = y_train[train_index].astype(int), y_train[test_index].astype(int) 

            # Decision Tree
            dt_tree = DecisionTree(max_depth=max_depth)
            dt_tree.fit(X_train_fold, y_train_fold)
            dt_train_y_pred = dt_tree.predict(X_train_fold)
            dt_test_y_pred = dt_tree.predict(X_test_fold)
            dt_train_accuracy, _ = dt_tree.accuracy(dt_train_y_pred, y_train_fold)
            dt_test_accuracy, _ = dt_tree.accuracy(dt_test_y_pred, y_test_fold)
            dt_train_accuracy_sum += dt_train_accuracy
            dt_test_accuracy_sum += dt_test_accuracy

            # Random Forest
            rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=42)
            rf_classifier.fit(X_train_fold, y_train_fold)
            rf_train_y_pred = rf_classifier.predict(X_train_fold)
            rf_test_y_pred = rf_classifier.predict(X_test_fold)
            rf_train_accuracy, _ = dt_tree.accuracy(rf_train_y_pred, y_train_fold)
            rf_test_accuracy, _ = dt_tree.accuracy(rf_test_y_pred, y_test_fold)
            rf_train_accuracy_sum += rf_train_accuracy
            rf_test_accuracy_sum += rf_test_accuracy
            
            # Append CV scores for this fold
            dt_cv_scores_for_depth.append(dt_test_accuracy)
            rf_cv_scores_for_depth.append(rf_test_accuracy)

        dt_train_accuracies.append(dt_train_accuracy_sum / kf.n_splits)
        dt_test_accuracies.append(dt_test_accuracy_sum / kf.n_splits)
        rf_train_accuracies.append(rf_train_accuracy_sum / kf.n_splits)
        rf_test_accuracies.append(rf_test_accuracy_sum / kf.n_splits)
        
        # Store average CV scores for this depth
        dt_cv_scores.append(np.mean(dt_cv_scores_for_depth))
        rf_cv_scores.append(np.mean(rf_cv_scores_for_depth))
        
        print(f"Max Depth: {max_depth}")
        print(f"Decision Tree - Train Accuracy: {dt_train_accuracy_sum / kf.n_splits:.4f}, Test Accuracy: {dt_test_accuracy_sum / kf.n_splits:.4f}")
        print(f"Random Forest - Train Accuracy: {rf_train_accuracy_sum / kf.n_splits:.4f}, Test Accuracy: {rf_test_accuracy_sum / kf.n_splits:.4f}")
        print("="*50)

        # Store best parameters based on test accuracy
        if best_dt_params is None or np.mean(dt_cv_scores_for_depth) > np.mean(dt_cv_scores[best_dt_params['max_depth']]):
            best_dt_params = {'max_depth': max_depth}
        if best_rf_params is None or np.mean(rf_cv_scores_for_depth) > np.mean(rf_cv_scores[best_rf_params['max_depth']]):
            best_rf_params = {'max_depth': max_depth}

    print("Decision Tree Cross-Validation Scores:", dt_cv_scores)
    print("Random Forest Cross-Validation Scores:", rf_cv_scores)
    print("Best Decision Tree Parameters:", best_dt_params)
    print("Best Random Forest Parameters:", best_rf_params)
    
    # Predict using best parameters on testing data
    best_dt_tree = DecisionTree(max_depth=best_dt_params['max_depth'])
    best_dt_tree.fit(X_train, y_train)
    best_dt_test_y_pred = best_dt_tree.predict(X_test)

    best_rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=best_rf_params['max_depth'], random_state=42)
    best_rf_classifier.fit(X_train, y_train)
    best_rf_test_y_pred = best_rf_classifier.predict(X_test)
    
    dt_test_accuracy, _ = best_dt_tree.accuracy(best_dt_test_y_pred, y_test)
    rf_test_accuracy, _ = best_dt_tree.accuracy(best_rf_test_y_pred, y_test)

    print("Decision Tree - Test Accuracy:", dt_test_accuracy)
    print("Random Forest - Test Accuracy:", rf_test_accuracy)
    
    # Plot train and test accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(dt_train_accuracies, label='DT Train Accuracy')
    plt.plot(dt_test_accuracies, label='DT Test Accuracy')
    plt.plot(rf_train_accuracies, label='RF Train Accuracy')
    plt.plot(rf_test_accuracies, label='RF Test Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title(f'Train and Test Accuracies for {dataset_name}')
    plt.legend()
    plt.savefig(f'C:/My_Project/train_test_accuracies_{dataset_name}.png')
    plt.show()
    
    plot_combined_confusion_matrix(y_test, best_dt_test_y_pred, best_rf_test_y_pred, dataset_name)
    classes, accuracies_dt, accuracies_rf = class_wise_accuracy(y_test, best_dt_test_y_pred, best_rf_test_y_pred)
    plot_class_wise_accuracy(classes, accuracies_dt, accuracies_rf, dataset_name)
    


# Call the function with the testing data
max_depth_range = range(1, 21)
# Processing First dataset
# Separate features (X) and labels (y)
train_data = pd.read_csv('C:/My_Project/Greek_dataset_1st/train.csv', header=None)
test_data = pd.read_csv('C:/My_Project/Greek_dataset_1st/test.csv', header=None)
X_train = np.asarray(train_data.iloc[:, :-1]).astype(float)
y_train = np.asarray(train_data.iloc[:, -1]).astype(int) 

X_test = np.asarray(test_data.iloc[:, :-1]).astype(float)
y_test = np.asarray(test_data.iloc[:, -1]).astype(int) 



model_with_cross_validation(X_train, y_train, X_test, y_test, max_depth_range, dataset_name="Greek Classification")
##########################################################################################
# Processing Second dataset
data = pd.read_csv('C:/My_Project/pca_output.csv')
X_train, X_test, y_train, y_test = data_split(data)
model_with_cross_validation(X_train, y_train,X_test, y_test, max_depth_range, dataset_name="GCDB")

# %%
