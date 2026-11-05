# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Program:

```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv("Placement_Data.csv")

# View first 5 rows
print("First 5 rows of the dataset:")
print(data.head())
data1 = data.copy()

# Dropping 'sl_no' (serial number) and 'salary' (not needed for predicting placement)
data1 = data1.drop(["sl_no", "salary"], axis=1)

print("\nData after dropping 'sl_no' and 'salary':")
print(data1.head())
print("\nChecking for missing values (True = missing):")
print(data1.isnull().any())

print("\nNumber of duplicate rows:")
print(data1.duplicated().sum())
cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
            "degree_t", "workex", "specialisation", "status"]

le = LabelEncoder()

for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])

print("\nData after Label Encoding:")
print(data1.head())
X = data1.iloc[:, :-1]
# y = 'status' column
y = data1["status"]

print("\nFeatures (X) sample:")
print(X.head())

print("\nTarget (y) sample:")
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
lr = LogisticRegression(solver="liblinear")

# Train the model
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("\nPredicted values (y_pred):")
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Classification Report: precision, recall, F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]

new_prediction = lr.predict(new_student)

print("\nPrediction for new student (0 = Not Placed, 1 = Placed):")
print(new_prediction[0])


```

## Output:

<img width="1208" height="315" alt="image" src="https://github.com/user-attachments/assets/956b7f1e-c1b3-4014-a454-54ba454f599b" />

<img width="1063" height="332" alt="image" src="https://github.com/user-attachments/assets/799c440d-e595-4cd3-8a94-c497425ce42f" />

<img width="952" height="353" alt="image" src="https://github.com/user-attachments/assets/fd7375f1-6813-4d71-969b-b5c59f3dd5d9" />

<img width="1090" height="396" alt="image" src="https://github.com/user-attachments/assets/8c1cfa08-250c-4d26-b886-af11aa265705" />

<img width="1032" height="337" alt="image" src="https://github.com/user-attachments/assets/6501e2c9-8300-42f3-9f13-f6d1d4393bc0" />

<img width="1017" height="417" alt="image" src="https://github.com/user-attachments/assets/c603bc4f-0b23-470f-aa89-edf0127dc52d" />

<img width="1282" height="367" alt="image" src="https://github.com/user-attachments/assets/0a604e27-5b4b-4c1f-88b7-4ced18338fcf" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
