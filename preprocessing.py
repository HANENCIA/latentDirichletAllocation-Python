# -*- coding: utf-8 -*-

import time
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def make_keyword_df(raw_csv_path, date_col_name, keyword_col_name):
    """
    make_keyword_df

    :param raw_csv_path:
    :param date_col_name:
    :param keyword_col_name:
    :return:
    """
    start_time = time.time()
    print("INFO: Making keyword dataframe from " + str(raw_csv_path))

    raw_df = pd.read_csv(raw_csv_path)

    keyword_df = raw_df.loc[:, [keyword_col_name]]
    keyword_df = keyword_df[keyword_col_name]

    date_df = raw_df.loc[:, [date_col_name]]
    # date_df[date_col_name] = date_df[date_col_name].astype(str).apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}")
    date_df[date_col_name] = date_df[date_col_name].astype(str).apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    keyword_df.index = date_df[date_col_name]

    end_time = time.time()
    print("INFO: completed! elapsed time " + str(round(end_time - start_time, 2)) + "s")

    return keyword_df


def remove_stopwords(keyword_df, stopwords_csv_path):
    """
    remove_stopwords

    :param keyword_df:
    :param stopwords_csv_path:
    :return:
    """
    start_time = time.time()
    print("INFO: Removing stopwords from : " + str(stopwords_csv_path))

    stopwords_lst = pd.read_csv(stopwords_csv_path, sep=",", encoding="utf-8")['STOPWORDS'].tolist()
    keyword_df = keyword_df.dropna().str.replace(",", " ")

    for idx in range(len(keyword_df)):
        tmp_lst = keyword_df[idx]
        if len(tmp_lst) == 0:
            continue
        tmp_lst = tmp_lst.split(" ")
        tmp_lst = [x for x in tmp_lst if x not in stopwords_lst]
        keyword_df[idx] = " ".join(tmp_lst)

    end_time = time.time()
    print("INFO: completed! elapsed time " + str(round(end_time - start_time, 2)) + "s")

    return keyword_df


def count_vectorizer(keyword_df, max_features=100000):
    """
    count_vectorizer

    :param keyword_df:
    :param max_features:
    :return: dtm => (document idx, term idx) count value
    """
    start_time = time.time()
    print("INFO: Vectorizing using count vectorizer")

    count_vector = CountVectorizer(max_features=max_features)
    keyword_lst = keyword_df.values
    count_dtm = count_vector.fit_transform(keyword_lst)

    end_time = time.time()
    print("INFO: completed! elapsed time " + str(round(end_time - start_time, 2)) + "s")

    return count_dtm, count_vector


def tfidf_vectorizer(keyword_df, ngram_max=1, max_df=0.6, min_df=1, max_features=100000):
    """
    tfidf_vectorizer

    :param keyword_df:
    :param ngram_max: only unigram (1, 1), unigram and bigram (1, 2), only bigram (2, 2)
    :param max_df: max document freq (int or float)
        0.5: ignore terms that appear in more than 50% of the documents
        25: ignore terms that appear in more than 25 documents
        1.0(default): ignore terms that appear in more than 100% of the documents (does not ignore any terms)
    :param min_df: min document freq (int or float)
        0.01: ignore terms that appear in less than 1% of the documents
        5: ignore terms that appear in less than 5 documents
        1(default): ignore terms that appear in less than 1 document (doesn't ignore any terms)
    :param max_features: max word count within a document
    :return: dtm => (document idx, term idx) tfidf value
    """
    start_time = time.time()
    print("INFO: Vectorizing using TF-IDF vectorizer")

    tfidf_vector = TfidfVectorizer(
        ngram_range=(1, ngram_max),
        max_df=max_df,
        min_df=min_df,
        max_features=max_features
    )

    keyword_lst = keyword_df.values
    tfidf_dtm = tfidf_vector.fit_transform(keyword_lst)

    end_time = time.time()
    print("INFO: completed! elapsed time " + str(round(end_time - start_time, 2)) + "s")

    return tfidf_dtm, tfidf_vector
