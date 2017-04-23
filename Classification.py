# coding:utf-8
from sklearn import svm
from sklearn.cross_validation import cross_val_score, KFold
import numpy as np


class IrisClassification:

    def __init__(self):
        self.read_data()

    # Read data and change to numpy array
    def read_data(self):

        file = open("iris.data")

        iris = file.read().split("\n")
        iris_data = []
        label = []

        # Input data to iris_data, iris_label
        for i in iris:
            iris_data.append(i.split(",")[:-1])
            label.append(i.split(",")[-1])

        iris_data = iris_data[:-2]
        label = label[:-2]
        iris_label = []

        # Change label to num
        for i in label:
            if i == 'Iris-setosa':
                iris_label.append(0)
            elif i == 'Iris-versicolor':
                iris_label.append(1)
            else:
                iris_label.append(2)

        learning_data = []
        for i in iris_data:
            temp = []
            for j in i:
                temp.append(float(j))
            learning_data.append(temp)
    
        self.data = np.array(learning_data)
        self.label = np.array(iris_label)


    # Create model
    def training(self):
        self.clf = svm.SVC(decision_function_shape='ovo', degree=3)
        self.clf.fit(self.data, self.label)


    # Predict data's label
    def predict_data(self, data):
        self.pred = self.clf.predict(data)

        if self.pred == 0:
            self.pred_name = 'Iris-setosa'
        elif self.pred == 1:
            self.pred_name = 'Iris-versicolor'
        else:
            self.pred_name = 'Iris-virginica'


    # Evaluate use 10 fold crossvalidation
    def evaluate(self):
        self.scores = cross_val_score(svm.SVC(decision_function_shape='ovo', degree=3), self.data, self.label, cv=10)
        self.score = 0

        for i in self.scores:
            self.score += i / len(self.scores)

        
