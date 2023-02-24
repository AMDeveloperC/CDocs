from preprocessing.pre_processor import PreProcessor
from supervised_learning.classification_models import Classification
from supervised_learning.supervised_learning import SupervisedLearning
from models.lsi_model import LSI_Model
from models.tf_idf_model import TfIdf_Model

pre_processor = PreProcessor()
documents_name = pre_processor.extract_documents_and_words("./docs/")
clean_corpus = pre_processor.preprocessing()

tf_idf_model = TfIdf_Model(clean_corpus)
(tf_idf, t_corpus, t_dictionary) = tf_idf_model.train_model()
lsi_model = LSI_Model(clean_corpus)
lsi_model.set_corpus(t_corpus)
(lsi, l_corpus, l_dictionary) = lsi_model.train_model()

labels = pre_processor.extract_labels_for_supervised_learning()
features = pre_processor.extract_features_for_supervised_learning(l_dictionary)

s_learning = SupervisedLearning(l_corpus, l_dictionary, clean_corpus, labels)
data_frame = s_learning.get_as_data_frame(features)
x_train, x_test, y_train, y_test = s_learning.split_dataset(['protect', 'algorithm'], 0.4)

classificator = Classification(x_train, x_test, y_train, y_test)
classificator.predict_svc()