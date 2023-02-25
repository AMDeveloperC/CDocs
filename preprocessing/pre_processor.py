from os import listdir
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from gensim.parsing.porter import PorterStemmer
from collections import defaultdict
import codecs
import pprint

class PreProcessor:
    def __del__(self):
        pass

    def __init__(self):
        self.documents_words_list = []
        self.documents_file_names = []
        self.texts = []
        self.labels = []
        self.en_stop_words = get_stop_words('en')

    def extract_documents_and_words(self, input_path):
        """
        Input  : the path on file system containing the documents to use
        Purpose: prepare a list of strings where each string corresponds to a document
        Output : a list of strings stored in self.documents_words_list
        """
        print("Reading folder: " + str(input_path))
        content = ""
        for document in listdir(input_path):
            self.documents_file_names.append(document)
            self.documents_file_names.sort()
        for document in self.documents_file_names:
            with codecs.open(str(input_path) + str(document), encoding = 'latin-1') as file_handler:
                for line in file_handler:
                    content = content + line
                    for i in ",][)(}{":
                        content = content.replace(i, ' ')
                self.documents_words_list.append(content)
                content = ""
        return self.documents_file_names
    
    def extract_labels_for_supervised_learning(self):
        self.documents_file_names.sort()
        i = 0
        while(i < len(self.documents_file_names)):
            self.labels.append(""+str(self.documents_file_names[i].split("_")[0]))
            i = i + 1
        return self.labels

    def extract_features_for_supervised_learning(self, dictionary):
        """
        Input  : the dictionary created with gensim
        Purpose: extract the elements and convert to a tuple
        """
        nums = []
        for i in range(len(dictionary)):
            nums.append(dictionary[i])
        return tuple(nums)

    def load_stop_words(self, stop_words_file):
        """
        Input  : a file containing one string per line
        Purpose: add new stop words
        Output : the self.en_stop_words list updated
        """
        with open(stop_words_file) as file_handler:
            for line in file_handler:
                words = line.split()
        self.en_stop_words = self.en_stop_words + words

    def stop_words_removal_and_stemming(self):
        """
        Purpose: execute stop words removal and stemming
        Output : a collection of words list (one per document)
        """
        p_stemmer = PorterStemmer()
        tokenizer = RegexpTokenizer(r'\w+')
        for w in self.documents_words_list:
            raw = w.lower()
            tokens = tokenizer.tokenize(raw)
            stopped_tokens = [t for t in tokens if not t in self.en_stop_words and not t.isdigit() and len(t) > 1]
            stemmed_tokens = [p_stemmer.stem(t) for t in stopped_tokens]
            self.texts.append(stemmed_tokens)

    def remove_words_only_once_appearing(self):
        """
        Purpose: remove low frequency words
        Output : updates the collection of words list
        """
        frequency = defaultdict(int)
        for text in self.texts:
            for token in text:
                frequency[token] += 1
        self.texts = [[token for token in text if frequency[token] > 1] for text in self.texts]

    def remove_shortest_words(self):
        """
        Purpose: remove words shorter than 3 characters
        Output : updates the collection of words list
        """
        self.texts = [[token for token in text if len(token) > 3] for text in self.texts]

    def preprocessing(self):
        """
        Purpose: executes preprocessing phase on the input documents
        Output : a list of cleaned and vectorized documents
        """
        self.remove_words_only_once_appearing()
        self.stop_words_removal_and_stemming()
        self.remove_shortest_words()
        return self.texts
    
    def clear(self):
        self.documents_file_names.clear()
        self.documents_words_list.clear()
        self.texts.clear()
        self.labels.clear()