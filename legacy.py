# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def warning_handler(*args, **kwargs):
    pass


warnings.warn = warning_handler

from datetime import datetime
import itertools
# from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfVectorizer


class Corpus:
    def __init__(self,
                 raw_csv_path,
                 raw_sep=",",
                 raw_enc="utf-8",
                 keyword_col_name="keyword",
                 date_col_name="date",
                 n_gram=1,
                 max_relative_frequency=1,
                 min_absolute_frequency=0,
                 max_features=100000):
        self._raw_csv_path = raw_csv_path
        self.date_col_name = date_col_name
        self._n_gram = n_gram
        self._max_relative_frequency = max_relative_frequency
        self._min_absolute_frequency = min_absolute_frequency
        self._max_features = max_features
        self.data_frame = pd.read_csv(raw_csv_path, sep=raw_sep, encoding=raw_enc)
        self.data_frame[date_col_name] = self.data_frame[date_col_name].astype(str).apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d").year)
        self.data_frame.fillna(' ')
        self.size = self.data_frame.count(0)[0]

        stop_words = []
        # stop_words = stopwords.words('english')
        vectorizer = TfidfVectorizer(ngram_range=(1, self._n_gram),
                                     max_df=self._max_relative_frequency,
                                     min_df=self._min_absolute_frequency,
                                     max_features=self._max_features,
                                     stop_words=stop_words)
        self.sklearn_vector_space = vectorizer.fit_transform(self.data_frame[keyword_col_name].tolist())
        self.gensim_vector_space = None
        vocab = vectorizer.get_feature_names()
        self.vocabulary = dict([(i, s) for i, s in enumerate(vocab)])

    def date(self, doc_id, date_col_name):
        return self.data_frame.iloc[doc_id][date_col_name]

    def word_for_id(self, word_id):
        return self.vocabulary.get(word_id)

    def doc_ids(self, date, date_col_name):
        return self.data_frame[self.data_frame[date_col_name] == date].index.tolist()


class TopicModel(object):

    def __init__(self, corpus):
        self.corpus = corpus
        self.date_col_name = corpus.date_col_name
        self.document_topic_matrix = None
        self.topic_word_matrix = None
        self.nb_topics = None

    def infer_topics(self, num_topics=None, **kwargs):
        pass

    def get_topics(self, num_words=None):
        frequency = self.topics_frequency()
        topic_lst = []

        for topic_id in range(self.nb_topics):
            word_list = []
            for weighted_word in self.top_words(topic_id, num_words):
                word_list.append(weighted_word[0])
            topic_lst.append((topic_id, frequency[topic_id], ", ".join(word_list)))

        return topic_lst

    def topic_frequency(self, topic, date=None):
        return self.topics_frequency(date=date)[topic]

    def topics_frequency(self, date=None):
        frequency = np.zeros(self.nb_topics)
        if date is None:
            ids = range(self.corpus.size)
        else:
            ids = self.corpus.doc_ids(date, self.date_col_name)

        for i in ids:
            topic = self.most_likely_topic_for_document(i)
            frequency[topic] += 1.0 / len(ids)
        return frequency

    def most_likely_topic_for_document(self, doc_id):
        weights = list(self.topic_distribution_for_document(doc_id))
        return weights.index(max(weights))

    def topic_distribution_for_document(self, doc_id):
        vector = self.document_topic_matrix[doc_id].toarray()
        return vector[0]

    def top_words(self, topic_id, num_words):
        vector = self.topic_word_matrix[topic_id]
        cx = vector.tocoo()
        weighted_words = [()] * len(self.corpus.vocabulary)
        for row, word_id, weight in itertools.zip_longest(cx.row, cx.col, cx.data):
            weighted_words[word_id] = (self.corpus.word_for_id(word_id), weight)
        weighted_words.sort(key=lambda x: x[1], reverse=True)
        return weighted_words[:num_words]


class LatentDirichletAllocation(TopicModel):
    def infer_topics(self, num_topics=None, max_iter=100, **kwargs):
        self.nb_topics = num_topics
        # lda_model = NMF(n_components=num_topics)
        lda_model = LDA(n_components=num_topics, learning_method='batch', max_iter=max_iter)
        topic_document = lda_model.fit_transform(self.corpus.sklearn_vector_space)

        self.topic_word_matrix = []
        self.document_topic_matrix = []
        vocabulary_size = len(self.corpus.vocabulary)
        row = []
        col = []
        data = []
        for topic_idx, topic in enumerate(lda_model.components_):
            for i in range(vocabulary_size):
                row.append(topic_idx)
                col.append(i)
                data.append(topic[i])
        self.topic_word_matrix = coo_matrix((data, (row, col)),
                                            shape=(self.nb_topics, len(self.corpus.vocabulary))).tocsr()
        row = []
        col = []
        data = []
        doc_count = 0
        for doc in topic_document:
            topic_count = 0
            for topic_weight in doc:
                row.append(doc_count)
                col.append(topic_count)
                data.append(topic_weight)
                topic_count += 1
            doc_count += 1
        self.document_topic_matrix = coo_matrix((data, (row, col)), shape=(self.corpus.size, self.nb_topics)).tocsr()


def main():
    raw_csv_path = "./data/sample.csv"
    raw_sep = ","
    raw_enc = "utf-8"

    keyword_col_name = "keyword"
    date_col_name = "date"

    tfidf_max_features = 100000
    tfidf_max_tf = 1.0
    tfidf_min_tf = 1
    tfidf_ngram = 2

    n_topics = 5
    n_iter = 10

    top_n_words = 10
    start_year = 2022
    end_year = 2022

    result_topic_words_csv_path = "./result/sample_legacy/topic_words.csv"
    result_topic_ratio_year_csv_path = "./result/sample_legacy/topic_ratio_year.csv"

    corpus = Corpus(raw_csv_path, raw_sep, raw_enc, keyword_col_name, date_col_name, tfidf_ngram, tfidf_max_tf,
                    tfidf_min_tf, tfidf_max_features)

    topic_model = LatentDirichletAllocation(corpus=corpus)

    topic_model.infer_topics(num_topics=n_topics, max_iter=n_iter)

    topic_words_lst = topic_model.get_topics(num_words=top_n_words)
    topic_words_df = pd.DataFrame(topic_words_lst, columns=["Topic No", "Frequency", "Words"])
    topic_words_df.to_csv(result_topic_words_csv_path, sep=",", encoding="utf-8-sig", index=False)

    topic_ratio_yr_lst = []
    for topic_id in range(topic_model.nb_topics):
        topic_ratio_yr_tmp = []
        for i in range(start_year, end_year + 1):
            topic_ratio_yr_tmp.append(topic_model.topic_frequency(topic_id, date=i))
        topic_ratio_yr_lst.append(topic_ratio_yr_tmp)

    year_topic_ratio_col_names = []
    for e in range(start_year, end_year + 1):
        year_topic_ratio_col_names.append(e)

    topic_ratio_yr_df = pd.DataFrame(topic_ratio_yr_lst, columns=year_topic_ratio_col_names)
    topic_ratio_yr_df.to_csv(result_topic_ratio_year_csv_path, sep=",", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
