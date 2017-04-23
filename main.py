# coding:utf-8
from Classification import IrisClassification

iris = IrisClassification()


# 10 fold crossvalidation
iris.evaluate()
print (iris.score)


# Create model and predict new data
iris.training()
iris.predict_data([4, 1, 5, 1])
print (iris.pred_name)

