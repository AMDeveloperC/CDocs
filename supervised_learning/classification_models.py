from supervised_learning.supervised_learning import SupervisedLearning
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
import pprint

class Classification:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def __del__(self):
        pass

    def predict_and_analyse(self, solver):
        self.lr_model = LogisticRegression(solver = solver)
        self.lr_model.fit(self.x_train, self.y_train)
        self.prediction = self.lr_model.predict(self.x_test)
        pprint.pprint(metrics.confusion_matrix(self.y_test, self.prediction))
        pprint.pprint(metrics.classification_report(self.y_test, self.prediction))

    def predict_svc(self):
        self.svc = SVC(gamma = 'auto')
        self.svc.fit(self.x_train, self.y_train)
        self.prediction = self.svc.predict(self.x_test)
        pprint.pprint(metrics.confusion_matrix(self.y_test, self.prediction))
        #pprint.pprint(metrics.classification_report(self.y_test, self.prediction))

    def save_classification_results(self, outout_file_name):
        with open(outout_file_name) as handler_file:
            for row in metrics.confusion_matrix(self.y_test, self.prediction):
                handler_file.write(row)
                handler_file.write("\n")

    def save_accuracy(self):
        pprint.pprint(metrics.accuracy_score(self.y_test, self.prediction))