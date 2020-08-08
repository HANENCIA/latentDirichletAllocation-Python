import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora
import gensim
import pyLDAvis.gensim
from sklearn.externals import joblib

if __name__ == "__main__":
    data = pd.read_csv('patent_KITECH.csv', encoding='utf-8').CONTENTS
    data3=[]
    for doc in data:
        data2 = word_tokenize(doc) # tokenizing & lowering
        data3.append(data2)

    dictionary = corpora.Dictionary(data3)
    corpus = [dictionary.doc2bow(word) for word in data3]

    lda = gensim.models.ldamulticore.LdaMulticore(corpus, batch=False, iterations=12, num_topics=10, id2word=dictionary, passes=1, workers=10)

    dictionary.save_as_text('dictionary.txt')
    corpora.MmCorpus.serialize('corpus.mm', corpus)
    joblib.dump(lda, 'model.pkl')

    lda_vis_ex51 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.show(lda_vis_ex51)