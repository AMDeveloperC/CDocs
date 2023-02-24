from gensim import models
from gensim import corpora

class LDA_Model:
    def __del__(self):
        pass

    def __init__(self, texts):
        self.texts = texts
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]

    def set_corpus(self, corpus):
        self.corpus = corpus

    def train_model(self):
        self.lda_model = models.LdaModel(self.corpus, id2word = self.dictionary, num_topics = 4, update_every = 3, chunksize = 10, passes = 1)
        self.topics = self.lda_model.print_topics(num_topics = 4, num_words = 10)
        self.l_corpus = self.lda_model[self.corpus]
        return self.lda_model, self.l_corpus, self.dictionary, self.topics

    def process_documents_query(self, unseen_docs):
        other_corpus = [self.dictionary.doc2bow(text) for text in unseen_docs]
        unseen_doc = other_corpus[0]
        self.vector = self.lda_model[unseen_doc]
        return self.vector

    def print_dict(self):
        for word in self.dictionary:
            print(str(word) + " " + str(self.dictionary[word]))

    def print_corpus(self):
        for doc in self.corpus:
            for t in doc:
                print(str(self.dictionary[t[0]]) + " " + str(t[1]) + " ")
            print("\n")