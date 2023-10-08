from preprocessing.pre_processor import PreProcessor
from supervised_learning.classification_models import Classification
from supervised_learning.supervised_learning import SupervisedLearning
from models.lsi_model import LSI_Model
from models.tf_idf_model import TfIdf_Model
import sys

pre_processor = PreProcessor()
pre_processor.extract_documents_and_words(sys.argv[1])
clean_corpus = pre_processor.preprocessing()

tf_idf_model = TfIdf_Model(clean_corpus)
(tf_idf, t_corpus, t_dictionary) = tf_idf_model.train_model()

labels = pre_processor.extract_labels_for_supervised_learning()
features = pre_processor.extract_features_for_supervised_learning(t_dictionary)

s_learning = SupervisedLearning(t_corpus, t_dictionary, clean_corpus, labels)
data_frame = s_learning.get_as_data_frame(features)
x_train, x_test, y_train, y_test = s_learning.split_dataset(sys.argv[2].split(" "), 0.4)

classificator = Classification(x_train, x_test, y_train, y_test)
print("Performing classification with SVC method: ")
classificator.train_svc()
print("Performing classification with logistic method: ")
classificator.train_logistic()
print("Performing classification with naive bayes classifier: ")
classificator.train_and_predict_naive_bayes()
