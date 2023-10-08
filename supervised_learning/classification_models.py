from supervised_learning.supervised_learning import SupervisedLearning
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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

    def train_logistic(self, solver = 'lbfgs', unseen_docs = None):
        if (unseen_docs is None):
            unseen_docs = self.x_test
        self.lr_model = LogisticRegression(solver = solver)
        self.prediction = self.lr_model.fit(self.x_train, self.y_train).predict(unseen_docs)
        print("Real labels: ");
        print(unseen_docs)
        print("Predicted labels: ")
        print(self.prediction)
        print("Misclassificated documents: " + str((self.y_test != self.prediction).sum()))

    def train_svc(self, unseen_docs = None):
        if (unseen_docs is None):
            unseen_docs = self.x_test
        self.svc = SVC(gamma = 'auto')
        self.prediction = self.svc.fit(self.x_train, self.y_train).predict(unseen_docs)
        print("Real labels: ");
        print(unseen_docs)
        print("Predicted labels: ")
        print(self.prediction)
        print("Misclassificated documents: " + str((self.y_test != self.prediction).sum()))

    def train_and_predict_naive_bayes(self, unseen_docs = None):
        if (unseen_docs is None):
            unseen_docs = self.x_test
        self.gnb = GaussianNB()
        self.prediction = self.gnb.fit(self.x_train, self.y_train).predict(unseen_docs)
        print("Real labels: ");
        print(unseen_docs)
        print("Predicted labels: ")
        print(self.prediction)
        print("Misclassificated documents: " + str((self.y_test != self.prediction).sum()))

    def print_metrix(self):
        pprint.pprint(metrics.confusion_matrix(self.y_test, self.prediction))
        pprint.pprint(metrics.classification_report(self.y_test, self.prediction))

    def save_classification_results(self, outout_file_name):
        with open(outout_file_name) as handler_file:
            for row in metrics.confusion_matrix(self.y_test, self.prediction):
                handler_file.write(row)
                handler_file.write("\n")

    def print_accuracy(self):
        pprint.pprint(metrics.accuracy_score(self.y_test, self.prediction))
