import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora
import pandas as pd
import gensim
from konlpy.tag import Okt
from collections import Counter
from itertools import chain
from sklearn.externals import joblib
import pyLDAvis.gensim

file = pd.ExcelFile('HAN.xlsx')

file.sheet_names

data = file.parse('Sheet1')

text = data['A']

noun = []
num=0
nlp = Okt()
for doc in text:
    print(num)
    noun.append(nlp.nouns(doc))
    if num%2==0:
        df=pd.DataFrame.from_records(zip(noun))
        writer = pd.ExcelWriter(str(num)+'.xlsx')
        df.to_excel(writer,'Sheet1')
        writer.save()
    num+=1

df=pd.DataFrame.from_records(zip(noun))
writer = pd.ExcelWriter(str(num)+'.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()

df

file = pd.ExcelFile('31.xlsx')
df = file.parse('Sheet1')

text = df[0]

text

# 1글자 명사와 공백 제거
def lengthTxt(func):
    newTxt = []
    for doc in func:
        filteredTxt = [w for w in doc if len(w) > 1]
        newTxt.append(filteredTxt)
    return newTxt
​
# 글자 대체 (토큰화 상태에서만 사용 가능)
def replaceTxt(func, old, new):
    newList = []
    for doc in func:
        newTxt = []
        for w in doc:
            newTxt.append(w.replace(old, new))
        newList.append(newTxt)
    return newList
​
# 토큰화 된 명사 합치기
def joinTxt(func):
    newTxt = []
    for doc in func:
        newTxt.append(" ".join(doc))
    return newTxt
​
# 명사 토큰화
def tokenTxt(func):
    newTxt = []
    for doc in func:
        newTxt.append(doc.split(' '))
    return newTxt
​
def replaceTxt2(func, old, new):
    newTxt = []
    for doc in func:
        newTxt.append(doc.replace(old, new))
    return newTxt

text=replaceTxt2(text, "]", '')
text=replaceTxt2(text, "[", '')
text=replaceTxt2(text, "'", '')
text=replaceTxt2(text, ",", '')

text

text2 = tokenTxt(text)

text2

# Stopwords 불러오기 (Stopwords는 그냥 엑셀로 작성, 열 이름만 STOPWORDS로 지정)
ko_stopwords_file = pd.ExcelFile('ko_stopwords.xlsx')
ko_stopwords = ko_stopwords_file.parse('Sheet1').STOPWORDS

# StopWords 제거
ko_stopped = text2

for w in ko_stopwords:
    ko_stopped = replaceTxt(ko_stopped, w, '')

# 1글자 명사와 공백 제거
ko_stopped = lengthTxt(ko_stopped)

# 빈도 상위 10000개의 명사를 나열하여 더 제거할 것이 있는지 확인
print(Counter(list(chain.from_iterable(ko_stopped))).most_common(100))

ko_stopped_join = joinTxt(ko_stopped)

ko_stopped_join

nltk.download()

# 영어
file_en = pd.ExcelFile('ENG.xlsx')
data_en = file_en.parse('Sheet1')
text_en = data_en['A']

stopWords = set(stopwords.words("english"))
stemmer = PorterStemmer()
en_stopped = []
num=0
p=re.compile('[a-zA-Z0-9\/]+')
for doc in text_en:
    doc1 = doc.lower()
    tokenizedWords= p.findall(doc1)
    stoppedWords = [w for w in tokenizedWords if w not in stopWords] # stopwords
    stemmedWords = [stemmer.stem(w) for w in tokenizedWords] # stemming
    en_stopped.append(stemmedWords)
    num+=1

en_stopped

dictionary = corpora.Dictionary(ko_stopped)
corpus = [dictionary.doc2bow(word) for word in ko_stopped]

lda = gensim.models.ldamulticore.LdaMulticore(corpus, batch=False, iterations=20, num_topics=4, id2word=dictionary, passes=1, workers=10)

dictionary.save_as_text('dictionary.txt')
corpora.MmCorpus.serialize('corpus.mm', corpus)
joblib.dump(lda, 'model.pkl')

dictionary = gensim.corpora.Dictionary.load_from_text('dictionary.txt')
corpus = gensim.corpora.MmCorpus('corpus.mm')
lda = joblib.load('model.pkl')

res = lda.get_document_topics(corpus)
tops, probs = zip(*res[0])

topicword = []
topicfreq = []
​
for index, topic in lda.show_topics(formatted=False, num_words=30):
    topicword.append([w[0] for w in topic])
    topicfreq.append([w[1] for w in topic])

char=[]
for i in range(len(probs)):
    char.append('topicword['+str(i)+']')
    char.append('topicfreq['+str(i)+']')
char = 'zip('+', '.join(char)+')'

wordfreq_zip=eval(char)

wordfreq = pd.DataFrame.from_records(wordfreq_zip)

writer = pd.ExcelWriter('wordfreq.xlsx', engine='xlsxwriter')
wordfreq.to_excel(writer,'Sheet1', index=False)
writer.save()

final_no = []
for i in range(len(probs)):
    final_no.append(str('Topic'+ str(i)))

final = pd.DataFrame.from_records(zip(final_no, probs, topicword), columns=['TOPIC', 'PROBABILITY', 'WORD'])

writer = pd.ExcelWriter('final.xlsx', engine='xlsxwriter')
final.to_excel(writer,'Sheet1', index=False)
writer.save()

for top in lda.print_topics(10):
    print(top)

lda_vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.show(lda_vis)

lda_vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(lda_vis, 'lda_vis.html')