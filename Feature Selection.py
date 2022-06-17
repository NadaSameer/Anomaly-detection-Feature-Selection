from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, metrics
import pandas as pd

"""
# this part of code is the reason why 24 features were chosen as it make a graph that indicates 
# how the performance goes from 16 to 31 features and apparently 24 gives the most performance

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt

fig1 = plot_sfs(sbs.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()
"""


def SVM():
    svmClassifier = svm.SVC(kernel='linear')
    svmClassifier.fit(xTrain, yTrain)
    svmPredicted_out = svmClassifier.predict(xTest)
    print("Accuracy:", metrics.accuracy_score(yTest, y_pred=svmPredicted_out))
    print("Precision:", metrics.precision_score(yTest, y_pred=svmPredicted_out))
    print("Recall", metrics.recall_score(yTest, y_pred=svmPredicted_out))


def Naive():
    naiveClassifier = GaussianNB()
    naiveClassifier.fit(xTrain, yTrain)
    naivePredicted_out = naiveClassifier.predict(xTest)
    print("Accuracy:", metrics.accuracy_score(yTest, y_pred=naivePredicted_out))
    print("Precision:", metrics.precision_score(yTest, y_pred=naivePredicted_out))
    print("Recall", metrics.recall_score(yTest, y_pred=naivePredicted_out))


def Logistic():
    logisticClassifier = LogisticRegression(solver='lbfgs', max_iter=3000)
    logisticClassifier.fit(xTrain, yTrain)
    logisticPredicted_out = logisticClassifier.predict(xTest)
    print("Accuracy:", metrics.accuracy_score(yTest, y_pred=logisticPredicted_out))
    print("Precision:", metrics.precision_score(yTest, y_pred=logisticPredicted_out))
    print("Recall", metrics.recall_score(yTest, y_pred=logisticPredicted_out))


def DecisionTree():
    dtClassifier = DecisionTreeClassifier()
    dtClassifier.fit(xTrain, yTrain)
    dtPredicted_out = dtClassifier.predict(xTest)
    print("Accuracy:", metrics.accuracy_score(yTest, y_pred=dtPredicted_out))
    print("Precision:", metrics.precision_score(yTest, y_pred=dtPredicted_out))
    print("Recall", metrics.recall_score(yTest, y_pred=dtPredicted_out))


data_file = pd.read_csv("data.csv")
x = data_file.drop('diagnosis', axis=1)
y = data_file.diagnosis
labelEncoder = LabelEncoder()
yEncoded = labelEncoder.fit_transform(y)

backwardWrapper = SequentialFeatureSelector(GaussianNB(), k_features=24, forward=False, floating=False, cv=0, scoring="precision")
backwardWrapper.fit(x, yEncoded)
selectedF = backwardWrapper.k_feature_names_
xSelected = pd.DataFrame(backwardWrapper.transform(x), columns=selectedF)
# print(xSelected.head())

xTrain, xTest, yTrain, yTest = train_test_split(xSelected, yEncoded, test_size=0.3)

# SVM
print("\nSvm")
SVM()

# Naive
print("\nNaive")
Naive()

# Logistic
print("\nLogistic Regression")
Logistic()

# Decision Tree
print("\nDecision Tree")
DecisionTree()

"""
   We have used the wrapper as it is a more realistic view of selecting features unlike the filter 
method that is quick but less accurate the wrapper method is more accurate especially the backward 
version as it takes all the feature in one go then starts to calculate the performance or accuracy  
that each subset gives removing redundant features as it goes giving us the most optimum features 
that are to be selected
"""