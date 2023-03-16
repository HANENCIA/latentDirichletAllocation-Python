# -*- coding: utf-8 -*-

from collections import Counter
import numpy as np


def get_keys(topic_matrix):
    """
    get_keys connects with get_topic_word(), get_topic_graph(), get_year_density_graph(), get_tsne_graph()
    :param topic_matrix:
    :return: keys => topic idx (n_topic) or topic categories, ex 0 ~29
    """
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys


def get_top_n_words(n_words, keys, document_term_matrix, n_topics, vectorizer):
    """
    get_top_n_words

    :param n_words: n issued words
    :param keys: topic index list
    :param document_term_matrix:
    :param n_topics:
    :param vectorizer
    :return: a list of n_topic strings, where each string contains the n most common words in a predicted category
    (or issued topics), in order
    """
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n_words:], 0)
        top_word_indices.append(top_n_word_indices)

    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1, document_term_matrix.shape[1]))
            temp_word_vector[:, index] = 1
            the_word = vectorizer.inverse_transform(temp_word_vector)[0][0]
            # topic_words.append(the_word.encode('ascii').decode('utf-8'))
            topic_words.append(the_word)
        top_words.append(" ".join(topic_words))

    return top_words


def keys_to_counts(keys, n_topics):
    """
    keys_to_counts get number of document (counts) classified in each topic (categories)
        connecting with get_topic_graph(), get_tsne_graph()
    :param keys:
    :param n_topics:
    :return:
    """
    count_pairs = Counter(keys).items()
    topics_lst = list(np.arange(0, n_topics))
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    categories_zero = set(topics_lst) - set(categories)
    for e in categories_zero:
        categories.append(e)
        counts.append(0)
    ret = (categories, counts)
    return ret