import pandas as pd
from konlpy.tag import Twitter

test = pd.read_csv('raw.csv', sep=',', index_col=None).내용

test2 = []
i=1
for value in test:
    nlp = Twitter()
    nouns = nlp.nouns(str(value))
    test3= " ".join(nouns)
    test2.append(test3)
    print(str(i)+"번 완료")
    i+=1

test4=pd.DataFrame(test2)

test4.to_csv('test3.csv', header=False, encoding='utf-8-sig')