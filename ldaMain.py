from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import itertools
import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np

MAX_FEATURES = 100000 #최대 단어 100,000개로 설정. 그 이상일 경우 늘릴것
FILE_PATH = 'lda_raw.xlsx'
FILE_SEPARATOR = ','
FILE_ENCODING = 'UTF-8'
MAX_TF = 0.6
MIN_TF = 1
num_topics = 15 #Topic의 수(K)
FIRST_YEAR = 1989
LAST_YEAR = 2018
FIRST_PERIOD = 1
LAST_PERIOD = 1

wordFreq=[]
wordFreqValue=[]
summaryValue_topic=[]
summaryValue_value=[]
summaryValue_word=[]

class Corpus:

    def __init__(self,
                 source_file_path,
                 n_gram=1,
                 max_relative_frequency=1,
                 min_absolute_frequency=0,
                 max_features=MAX_FEATURES):

        self._source_file_path = source_file_path
        self._n_gram = n_gram
        self._max_relative_frequency = max_relative_frequency
        self._min_absolute_frequency = min_absolute_frequency

        self.max_features = max_features
        self.data_frame = pd.ExcelFile(source_file_path).parse('Sheet1')
        # self.data_frame = pd.read_csv(source_file_path, sep=FILE_SEPARATOR, encoding=FILE_ENCODING) #파일 형태: 쉼표로 분리된 UTF-8로 인코딩 된 CSV(TXT) 파일
                                                                                    #파일 Header: DATE(년), ABSTRACT(초록)
        self.data_frame.fillna(' ') #결측치는 공백 처리
        self.size = self.data_frame.count(0)[0]

        stop_words = []
        # stop_words = stopwords.words('english') #영어 불용어 처리 할 경우 주석 해제, 이미 사전에 전처리했으므로 여기서는 생략
        vectorizer = TfidfVectorizer(ngram_range=(1, n_gram),
                                     max_df=max_relative_frequency,
                                     min_df=min_absolute_frequency,
                                     max_features=self.max_features,
                                     stop_words=stop_words)
        self.sklearn_vector_space = vectorizer.fit_transform(self.data_frame['CONTENTS'].tolist())
        self.gensim_vector_space = None
        vocab = vectorizer.get_feature_names()
        self.vocabulary = dict([(i, s) for i, s in enumerate(vocab)])

    def date(self, doc_id):
        return self.data_frame.iloc[doc_id]['DATE']

    def word_for_id(self, word_id):
        return self.vocabulary.get(word_id)

    def doc_ids(self, date):
        return self.data_frame[self.data_frame['DATE'] == date].index.tolist()

    def doc_ids2(self, date):
        return self.data_frame[self.data_frame['DATE2'] == date].index.tolist()


class TopicModel(object):

    def __init__(self, corpus):
        self.corpus = corpus
        self.document_topic_matrix = None
        self.topic_word_matrix = None
        self.nb_topics = None
        self.nb_topics2 = None

    def infer_topics(self, num_topics=None, **kwargs):
        pass

    def print_topics(self, num_words=None):
        frequency = self.topics_frequency()
        topic_list = []
        for topic_id in range(self.nb_topics):
            word_list = []
            word_list2 = []
            for weighted_word in self.top_words(topic_id, num_words):
                word_list.append(weighted_word[0])
                word_list2.append(weighted_word[1])
            topic_list.append((topic_id, frequency[topic_id], word_list))
            wordFreq.append(word_list)
            wordFreq.append(word_list2)
            # pd.DataFrame.from_records(zip(word_list, word_list2)).to_csv('WORD_FREQUENCY_'+str(topic_id)+'.csv', sep=',', header=True, index=False, encoding='utf-8-sig')
        for topic_id, frequency, topic_desc in topic_list:
            summaryValue_topic.append('topic %d' % topic_id)
            summaryValue_value.append('%f' % frequency)
            summaryValue_word.append('%s'% ', '.join(topic_desc))
            # summaryValue.append('topic %d, %f, %s' % (topic_id, frequency, ', '.join(topic_desc))) #TOPIC 출력


    def topic_frequency(self, topic, date=None):
        return self.topics_frequency(date=date)[topic]

    def topic_frequency2(self, topic, date=None):
        return self.topics_frequency2(date=date)[topic]

    def topics_frequency(self, date=None):
        frequency = np.zeros(self.nb_topics)
        if date is None:
            ids = range(self.corpus.size)
        else:
            ids = self.corpus.doc_ids(date)

        for i in ids:
            topic = self.most_likely_topic_for_document(i)
            frequency[topic] += 1.0 / len(ids)
        return frequency

    def topics_frequency2(self, date=None):
        frequency = np.zeros(self.nb_topics)
        if date is None:
            ids = range(self.corpus.size)
        else:
            ids = self.corpus.doc_ids2(date)

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
    def infer_topics(self, num_topics=None, **kwargs):
        self.nb_topics = num_topics
        lda_model = NMF(n_components=num_topics)
        # lda_model = LDA(n_topics=num_topics, learning_method='batch', max_iter=100) #LDA: max_iter='Gibbs sampling 횟수'
        topic_document = lda_model.fit_transform(self.corpus.sklearn_vector_space) #NMF: LDA 결과가 좋지 않을 때 보완적으로 사용
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
        self.document_topic_matrix = coo_matrix((data, (row, col)),
                                                shape=(self.corpus.size, self.nb_topics)).tocsr()


corpus = Corpus(source_file_path=FILE_PATH, max_relative_frequency=MAX_TF, min_absolute_frequency=MIN_TF)

topic_model = LatentDirichletAllocation(corpus=corpus)

topic_model.infer_topics(num_topics=num_topics)
topic_model.print_topics(num_words=100) #각 주제(topic)별 단어 20개 출력 (num_words='출력할 단어 갯수')

yearValue = []
yearTopicValue = []
periodValue = []
periodTopicValue = []

for i in range (FIRST_YEAR, LAST_YEAR+1):
    yearValue.append(i)

for i in range(FIRST_PERIOD, LAST_PERIOD + 1):
    periodValue.append(i)

for topic_id in range(topic_model.nb_topics):

    for i in range(FIRST_YEAR, LAST_YEAR+1):
        yearTopicValue.append(topic_model.topic_frequency(topic_id, date=i))

    for i in range(FIRST_PERIOD, LAST_PERIOD+1):
        periodTopicValue.append(topic_model.topic_frequency2(topic_id, date=i))

yearTopicValue = np.reshape(yearTopicValue, (num_topics, LAST_YEAR-FIRST_YEAR+1)).T
periodTopicValue = np.reshape(periodTopicValue, (num_topics, LAST_PERIOD-FIRST_PERIOD+1)).T

topicyear=pd.DataFrame(yearTopicValue, index=yearValue)
topicPeriod = pd.DataFrame(periodTopicValue, index=periodValue)
summaryValue=zip(summaryValue_topic, summaryValue_value, summaryValue_word)

pd.DataFrame.from_records(wordFreq).to_csv(str(num_topics)+'_WORDFREQ.csv', sep=',', header=False, index=False, encoding='utf-8-sig')
topicyear.to_csv(str(num_topics)+'_YEAR.csv', sep=',', header=True, index=True, encoding='utf-8-sig')
topicPeriod.to_csv(str(num_topics)+'_PERIOD.csv', sep=',', header=True, index=True, encoding='utf-8-sig')
pd.DataFrame.from_records(summaryValue).to_csv(str(num_topics)+'_SUMMARY.csv', sep=',', header=False, index=False, encoding='utf-8-sig')