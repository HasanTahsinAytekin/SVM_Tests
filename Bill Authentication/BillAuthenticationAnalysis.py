import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Original data: https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data
bankdata = pd.read_csv("Data\\BankNote_Authentication.csv")

# Dataset'leri düzenl
X = bankdata.drop('Class', axis=1)  
y = bankdata['Class']

# Eğitim ve test verilerini oluştur
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# SVC: Doğrusal
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# SVC: Polinom
svclassifier_poly = SVC(kernel='poly', degree=2)
svclassifier_poly.fit(X_train, y_train)

y_pred = svclassifier_poly.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# SVC: RBF
svclassifier_rbf = SVC(kernel='rbf')
svclassifier_rbf.fit(X_train, y_train)

y_pred = svclassifier_rbf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
