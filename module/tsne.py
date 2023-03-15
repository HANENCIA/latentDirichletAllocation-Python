# -*- coding: utf-8 -*-

from bokeh.models import Label
from bokeh.plotting import figure, output_file, show, save
from matplotlib import font_manager, rc
import numpy as np
import os
import platform
from sklearn.manifold import TSNE
import time


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


def get_mean_topic_vectors(keys, dimension, n_topics):
    """
    get_mean_topic_vectors makes centroid vector from each topic

    :param keys:
    :param dimension: target dimension reduction for vector (default: 2)
    :param n_topics:
    """
    mean_topic_vectors = []
    for t in range(n_topics):
        articles_in_that_topic = []
        for i in range(len(keys)):
            if keys[i] == t:
                articles_in_that_topic.append(dimension[i])

        articles_in_that_topic = np.vstack(articles_in_that_topic)
        mean_article_in_that_topic = np.mean(articles_in_that_topic, axis=0)
        mean_topic_vectors.append(mean_article_in_that_topic)
    return mean_topic_vectors


def make_tsne_graph(keyword_dtm, topic_matrix, vectorizer, n_topics, n_components, perplexity, learning_rate, n_iter,
                    random_state, angle, n_jobs, dest_path):
    """
    make_tsne_graph

    :param keyword_dtm: tfidf_dtm or count_dtm
    :param topic_matrix: lda_topic_matrix
    :param vectorizer: tfidf_vector or count_vector
    :param n_topics: number of topics
    :param n_components: target dimension reduction (default: 2)
    :param perplexity: considerable point count for learning, 5-50
    :param learning_rate: 10-1000
    :param n_iter: number of iterations
    :param random_state:
    :param angle: angular size of a distant node as measured from a point, 0.2-0.8 (default: method='barnes_hut')
        angle < 0.2 : calculation speed higher
        angle > 0.8 : error increasing
    :param n_jobs: number of CPU core (default: -1)
    :param dest_path
    :return:
    """
    start_time = time.time()
    print(f"INFO: Saving T-SNE graph to {dest_path}")

    if platform.system() == "Windows":
        font_name = font_manager.FontProperties(
            fname=str(os.path.join(os.environ["WINDIR"], "Fonts")) + "/malgun.ttf").get_name()
        rc('font', family=font_name)
    elif platform.system() == "Darwin":
        rc("font", family="AppleGothic")

    "T-SNE colormap"
    colormap = np.array([
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"])

    colormap = colormap[:n_topics]

    tsne_lda_model = TSNE(
        n_components=n_components, perplexity=perplexity, learning_rate=learning_rate,
        n_iter=n_iter, verbose=1, random_state=random_state, angle=angle, n_jobs=n_jobs)

    tsne_lda_vectors = tsne_lda_model.fit_transform(topic_matrix)
    keys = get_keys(topic_matrix)
    top_3_words_lda = get_top_n_words(3, keys, keyword_dtm, n_topics, vectorizer)
    lda_mean_topic_vectors = get_mean_topic_vectors(
        keys=keys, dimension=tsne_lda_vectors, n_topics=n_topics)

    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), plot_width=1500, plot_height=1500)
    plot.scatter(x=tsne_lda_vectors[:, 0], y=tsne_lda_vectors[:, 1], color=colormap[keys])

    for t in range(n_topics):
        label = Label(
            x=lda_mean_topic_vectors[t][0],
            y=lda_mean_topic_vectors[t][1],
            text=top_3_words_lda[t],
            text_color=colormap[t]
        )
        plot.add_layout(label)

    output_file(filename=dest_path)
    save(plot)

    end_time = time.time()
    print(f"INFO: Completed. Elapsed Time: {round(end_time - start_time, 2)}s")
