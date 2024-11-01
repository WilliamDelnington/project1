import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("E:/Works/AI/_torch_/framingham.csv")
data = data.dropna()
data = data.head(100)
X_vals = data[["age"]]
Y_vals = data[["heartRate"]]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_vals, Y_vals, test_size=0.3, random_state=10)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
img_reg = classifier.fit(X_train_scaled, Y_train)


# from sklearn.metrics import confusion_matrix, accuracy_score
# Y_pred = classifier.predict(X_test)
# cm = confusion_matrix(Y_test, Y_pred)

# print(cm)
# print(accuracy_score(Y_test, Y_pred))