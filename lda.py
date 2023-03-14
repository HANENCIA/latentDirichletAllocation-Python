# -*- coding: utf-8 -*-

import time
import pandas as pd
import pickle
from sklearn.decomposition import LatentDirichletAllocation

import common


def topic_modeling(keyword_dtm, n_topics, learning_method="batch", max_iter=100, random_state=0, n_jobs=-1):
    """
    topic_modeling

    :param keyword_dtm:
    :param n_topics: number of topics
    :param learning_method: (default: batch (Use all training data))
    :param max_iter: number of iteration
    :param random_state: random seed (generally 0)
    :param n_jobs: number of CPU core (default: -1)
    :return: lda_model, lda_topic_matrix
    """
    start_time = time.time()
    print("INFO: Topic modeling")

    lda_model = LatentDirichletAllocation(
        n_components=n_topics, learning_method=learning_method, max_iter=max_iter,
        random_state=random_state, verbose=1, n_jobs=n_jobs)

    lda_topic_matrix = lda_model.fit_transform(keyword_dtm)

    end_time = time.time()
    print("INFO: completed! elapsed time " + str(round(end_time - start_time, 2)) + "s")

    return lda_model, lda_topic_matrix


def get_topic_words(topic_matrix, keyword_dtm, n_words, n_topics, vectorizer, dest_path):
    """
    get_topic_words

    :param topic_matrix:
    :param keyword_dtm:
    :param n_words:
    :param n_topics:
    :param vectorizer:
    :param dest_path
    """
    start_time = time.time()
    print("INFO: Saving topic data into " + str(dest_path))

    keys = common.get_keys(topic_matrix)

    top_n_words_lda = common.get_top_n_words(n_words=n_words, keys=keys, document_term_matrix=keyword_dtm,
                                             n_topics=n_topics, vectorizer=vectorizer)

    result_lst = []
    for i in range(len(top_n_words_lda)):
        print(f"Topic {(i + 1)}: {top_n_words_lda[i]}")
        result_lst.append([f"Topic {(i + 1)}", top_n_words_lda[i]])

    result_df = pd.DataFrame(data=result_lst, columns=["Topic No.", "Topics"])
    result_df.to_csv(dest_path, sep=",", encoding="utf-8-sig", index=False)

    end_time = time.time()
    print("INFO: completed! elapsed time " + str(round(end_time - start_time, 2)) + "s")


def get_topic_distribution(topic_matrix):
    """
    get_topic_distribution

    :param topic_matrix:
    """
    topic_no: list = topic_matrix.argmax(axis=1)
    topic_no = [x + 1 for x in topic_no]
    topic_dist: list = topic_matrix.max(axis=1)

    topic_df = pd.DataFrame(list(zip(topic_no, topic_dist)), columns=["topic_no", "topic_dist"])

    return topic_df


def get_topic_distribution_top_n(raw_csv_path, lda_model_path, lda_topic_matrix_path, top_n, dest_path):
    """
    get_topic_distribution_top_n

    :param raw_csv_path:
    :param lda_model_path:
    :param lda_topic_matrix_path:
    :param top_n:
    :param dest_path
    """
    raw_df = pd.read_csv(raw_csv_path)

    lda_model, lda_topic_matrix = load_lda_model(
        model_path=lda_model_path,
        topic_matrix_path=lda_topic_matrix_path
    )

    topic_df = get_topic_distribution(lda_topic_matrix)

    result_df = pd.concat([raw_df, topic_df], axis=1)
    result_df = result_df.sort_values(by=["topic_no", "topic_dist"], ascending=[True, False])
    result_df = result_df.groupby('topic_no').head(top_n)

    result_df.to_csv(dest_path, sep=",", encoding="utf-8-sig", index=False)


def load_lda_model(model_path, topic_matrix_path):
    """
    load_lda_model

    :param model_path:
    :param topic_matrix_path:
    """
    start_time = time.time()
    print("INFO: Loading LDA model from " + str(model_path))

    with open(model_path, 'rb') as f:
        lda_model = pickle.load(f)

    end_time = time.time()
    print("INFO: completed! elapsed time " + str(round(end_time - start_time, 2)) + "s")

    print("INFO: Loading topic matrix from " + str(topic_matrix_path))

    with open(topic_matrix_path, 'rb') as f:
        lda_topic_matrix = pickle.load(f)

    end_time = time.time()
    print("INFO: completed! elapsed time " + str(round(end_time - start_time, 2)) + "s")

    return lda_model, lda_topic_matrix


def save_lda_model(lda_model, lda_topic_matrix, model_path, topic_matrix_path):
    """
    save_lda_model

    :param lda_model:
    :param lda_topic_matrix:
    :param model_path:
    :param topic_matrix_path:
    """
    start_time = time.time()
    print("INFO: Saving LDA model to " + str(model_path))

    with open(model_path, 'wb') as f:
        pickle.dump(lda_model, f)

    end_time = time.time()
    print("INFO: completed! elapsed time " + str(round(end_time - start_time, 2)) + "s")

    print("INFO: Saving topic matrix to " + str(topic_matrix_path))

    with open(topic_matrix_path, 'wb') as f:
        pickle.dump(lda_topic_matrix, f)

    end_time = time.time()
    print("INFO: completed! elapsed time " + str(round(end_time - start_time, 2)) + "s")
