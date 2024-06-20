# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.svm import SVC
# # from sklearn.neighbors import KNeighborsClassifier
# # from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # data = pd.read_csv('cdd.csv')
# # X = data.drop('Class', axis=1)
# # y = data['Class']
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # random_forest_model = RandomForestClassifier(random_state=42)
# # random_forest_model.fit(X_train, y_train)
# # y_pred_rf = random_forest_model.predict(X_test)

# # accuracy_rf = accuracy_score(y_test, y_pred_rf)
# # conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
# # classification_rep_rf = classification_report(y_test, y_pred_rf)
# # print("Random Forest Model:")
# # print(f"Accuracy: {accuracy_rf:.2f}")
# # print(f"Confusion Matrix:\n{conf_matrix_rf}")
# # print(f"Classification Report:\n{classification_rep_rf}")
# # print("-" * 50)
# # random_forest_model = RandomForestClassifier(random_state=42)
# # random_forest_model.fit(X_train, y_train)
# # y_pred_rf = random_forest_model.predict(X_test)

# # accuracy_rf = accuracy_score(y_test, y_pred_rf)
# # conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
# # classification_rep_rf = classification_report(y_test, y_pred_rf)
# # print("Random Forest Model:")
# # print(f"Accuracy: {accuracy_rf:.2f}")
# # print(f"Confusion Matrix:\n{conf_matrix_rf}")
# # print(f"Classification Report:\n{classification_rep_rf}")
# # print("-" * 50)

# # logistic_regression_model = LogisticRegression(random_state=42)
# # logistic_regression_model.fit(X_train, y_train)
# # y_pred_lr = logistic_regression_model.predict(X_test)

# # accuracy_lr = accuracy_score(y_test, y_pred_lr)
# # conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
# # classification_rep_lr = classification_report(y_test, y_pred_lr)
# # print("Logistic Regression Model:")
# # print(f"Accuracy: {accuracy_lr:.2f}")
# # print(f"Confusion Matrix:\n{conf_matrix_lr}")
# # print(f"Classification Report:\n{classification_rep_lr}")
# # print("-" * 50)

# # svm_model = SVC(random_state=42)
# # svm_model.fit(X_train, y_train)
# # y_pred_svm = svm_model.predict(X_test)

# # accuracy_svm = accuracy_score(y_test, y_pred_svm)
# # conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
# # classification_rep_svm = classification_report(y_test, y_pred_svm)
# # print("Support Vector Machine (SVM) Model:")
# # print(f"Accuracy: {accuracy_svm:.2f}")
# # print(f"Confusion Matrix:\n{conf_matrix_svm}")
# # print(f"Classification Report:\n{classification_rep_svm}")
# # print("-" * 50)

# # knn_model = KNeighborsClassifier()
# # knn_model.fit(X_train, y_train)
# # y_pred_knn = knn_model.predict(X_test)
# # accuracy_knn = accuracy_score(y_test, y_pred_knn)
# # conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
# # classification_rep_knn = classification_report(y_test, y_pred_knn)
# # print("K-Nearest Neighbors (KNN) Model:")
# # print(f"Accuracy: {accuracy_knn:.2f}")
# # print(f"Confusion Matrix:\n{conf_matrix_knn}")
# # print(f"Classification Report:\n{classification_rep_knn}")

# # # Taking input from the user
# # print("Please enter values for the features to predict the class:")
# # feature_names = X.columns.tolist()
# # user_input = []
# # for feature in feature_names:
# #     value = float(input(f"Enter value for {feature}: "))
# #     user_input.append(value)

# # # Predicting using the trained model
# # predicted_class = random_forest_model.predict([user_input])
# # print(f"The predicted class is: {predicted_class[0]} So it is Fraud ...



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # Load dataset
data = pd.read_csv('cdd.csv')
X = data.drop('Class', axis=1)
y = data['Class']

# # Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Random Forest Model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)
print("Random Forest Model:")
print(f"Accuracy: {accuracy_rf:.2f}")
print(f"Confusion Matrix:\n{conf_matrix_rf}")
print(f"Classification Report:\n{classification_rep_rf}")
print("-" * 50)

# # Logistic Regression Model
logistic_regression_model = LogisticRegression(random_state=42)
logistic_regression_model.fit(X_train, y_train)
y_pred_lr = logistic_regression_model.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
classification_rep_lr = classification_report(y_test, y_pred_lr)
print("Logistic Regression Model:")
print(f"Accuracy: {accuracy_lr:.2f}")
print(f"Confusion Matrix:\n{conf_matrix_lr}")
print(f"Classification Report:\n{classification_rep_lr}")
print("-" * 50)

# # Support Vector Machine (SVM) Model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
classification_rep_svm = classification_report(y_test, y_pred_svm)
print("Support Vector Machine (SVM) Model:")
print(f"Accuracy: {accuracy_svm:.2f}")
print(f"Confusion Matrix:\n{conf_matrix_svm}")
print(f"Classification Report:\n{classification_rep_svm}")
print("-" * 50)

# # K-Nearest Neighbors (KNN) Model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
classification_rep_knn = classification_report(y_test, y_pred_knn)
print("K-Nearest Neighbors (KNN) Model:")
print(f"Accuracy: {accuracy_knn:.2f}")
print(f"Confusion Matrix:\n{conf_matrix_knn}")
print(f"Classification Report:\n{classification_rep_knn}")

# # Taking input from the user
print("Please enter values for the features to predict the class:")
feature_names = X.columns.tolist()
user_input = []
for feature in feature_names:
    value = float(input(f"Enter value for {feature}: "))
    user_input.append(value)

# # Predicting using the trained model
predicted_class = random_forest_model.predict([user_input])
print(f"The predicted class is: {predicted_class[0]}")

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import tkinter as tk
# from tkinter import ttk, messagebox

# # Load dataset
# data = pd.read_csv('cdd.csv')
# X = data.drop('Class', axis=1)
# y = data['Class']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train Random Forest Model
# random_forest_model = RandomForestClassifier(random_state=42)
# random_forest_model.fit(X_train, y_train)

# # Logistic Regression Model
# logistic_regression_model = LogisticRegression(random_state=42)
# logistic_regression_model.fit(X_train, y_train)

# # Support Vector Machine (SVM) Model
# svm_model = SVC(random_state=42)
# svm_model.fit(X_train, y_train)

# # K-Nearest Neighbors (KNN) Model
# knn_model = KNeighborsClassifier()
# knn_model.fit(X_train, y_train)

# # Create the main application window
# root = tk.Tk()
# root.title("Machine Learning Model Predictions")

# # Create input fields for features
# feature_names = X.columns.tolist()
# entries = {}

# for feature in feature_names:
#     row = tk.Frame(root)
#     label = tk.Label(row, width=22, text=feature + ": ", anchor='w')
#     entry = tk.Entry(row)
#     row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
#     label.pack(side=tk.LEFT)
#     entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
#     entries[feature] = entry

# # Function to make predictions and display results
# def make_prediction():
#     user_input = []
#     try:
#         for feature in feature_names:
#             value = float(entries[feature].get())
#             user_input.append(value)
#     except ValueError:
#         messagebox.showerror("Input Error", "Please enter valid numeric values for all features.")
#         return
    
#     predicted_class = random_forest_model.predict([user_input])[0]
    
#     accuracy_rf = accuracy_score(y_test, random_forest_model.predict(X_test))
#     conf_matrix_rf = confusion_matrix(y_test, random_forest_model.predict(X_test))
#     classification_rep_rf = classification_report(y_test, random_forest_model.predict(X_test))

#     messagebox.showinfo(
#         "Prediction Result",
#         f"Predicted Class: {predicted_class}\n\nRandom Forest Model Performance:\n"
#         f"Accuracy: {accuracy_rf:.2f}\nConfusion Matrix:\n{conf_matrix_rf}\n"
#         f"Classification Report:\n{classification_rep_rf}"
#     )

# # Create the predict button
# predict_button = ttk.Button(root, text="Predict", command=make_prediction)
# predict_button.pack(side=tk.BOTTOM, padx=5, pady=5)

# # Run the GUI event loop
# root.mainloop()


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import tkinter as tk
# from tkinter import ttk, messagebox

# # Load dataset
# data = pd.read_csv('cdd.csv')
# X = data.drop('Class', axis=1)
# y = data['Class']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train Random Forest Model
# random_forest_model = RandomForestClassifier(random_state=42)
# random_forest_model.fit(X_train, y_train)

# # Logistic Regression Model
# logistic_regression_model = LogisticRegression(random_state=42)
# logistic_regression_model.fit(X_train, y_train)

# # Support Vector Machine (SVM) Model
# svm_model = SVC(random_state=42)
# svm_model.fit(X_train, y_train)

# # K-Nearest Neighbors (KNN) Model
# knn_model = KNeighborsClassifier()
# knn_model.fit(X_train, y_train)

# # Create the main application window
# root = tk.Tk()
# root.title("Machine Learning Model Predictions")

# # Create input fields for features
# feature_names = X.columns.tolist()
# entries = {}

# for feature in feature_names:
#     row = tk.Frame(root)
#     label = tk.Label(row, width=22, text=feature + ": ", anchor='w')
#     entry = tk.Entry(row)
#     row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
#     label.pack(side=tk.LEFT)
#     entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
#     entries[feature] = entry

# # Pre-fill sample input values for easier testing
# sample_inputs = {
#     'Age': 25,
#     'Income': 55000,
#     'CreditScore': 700,
#     'LoanAmount': 15000,
#     'YearsAtJob': 3
# }

# for feature, value in sample_inputs.items():
#     if feature in entries:
#         entries[feature].insert(0, str(value))

# # Function to make predictions and display results
# def make_prediction():
#     user_input = []
#     try:
#         for feature in feature_names:
#             value = float(entries[feature].get())
#             user_input.append(value)
#     except ValueError:
#         messagebox.showerror("Input Error", "Please enter valid numeric values for all features.")
#         return
    
#     predicted_class = random_forest_model.predict([user_input])[0]
    
#     accuracy_rf = accuracy_score(y_test, random_forest_model.predict(X_test))
#     conf_matrix_rf = confusion_matrix(y_test, random_forest_model.predict(X_test))
#     classification_rep_rf = classification_report(y_test, random_forest_model.predict(X_test))

#     messagebox.showinfo(
#         "Prediction Result",
#         f"Predicted Class: {predicted_class}\n\nRandom Forest Model Performance:\n"
#         f"Accuracy: {accuracy_rf:.2f}\nConfusion Matrix:\n{conf_matrix_rf}\n"
#         f"Classification Report:\n{classification_rep_rf}"
#     )

# # Create the predict button
# predict_button = ttk.Button(root, text="Predict", command=make_prediction)
# predict_button.pack(side=tk.BOTTOM, padx=5, pady=5)

# # Run the GUI event loop
# root.mainloop()
