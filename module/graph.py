# -*- coding: utf-8 -*-

import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import os
import numpy as np
import pandas as pd
import platform
import seaborn as sb

from . import common


def make_topic_graph(topic_matrix, keyword_dtm, n_topics, vectorizer, dest_path):
    """
    make_topic_graph makes word frequency bar chart with each topic

    :param topic_matrix:
    :param keyword_dtm:
    :param n_topics:
    :param vectorizer:
    :param dest_path:
    """
    start_time = time.time()
    print("INFO: Saving Num of Articles graph to " + str(dest_path))

    if platform.system() == "Windows":
        font_name = font_manager.FontProperties(
            fname=str(os.path.join(os.environ["WINDIR"], "Fonts")) + "/malgun.ttf").get_name()
        rc('font', family=font_name)
    elif platform.system() == "Darwin":
        rc("font", family="AppleGothic")

    keys = common.get_keys(topic_matrix)
    categories, counts = common.keys_to_counts(keys)
    top_3_words = common.get_top_n_words(3, keys, keyword_dtm, n_topics, vectorizer)
    labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in categories]

    fig, ax = plt.subplots(figsize=(50, 30))
    ax.bar(categories, counts)
    ax.set_title('LDA issue keyword counts with topic', fontsize=14, fontweight='bold')
    ax.set_xticks(categories)
    ax.set_xticklabels(labels, fontsize=8, rotation=45)
    ax.set_ylabel('Number of newspaper', fontsize=12)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.savefig(dest_path, dpi=300)

    end_time = time.time()
    print(f"INFO: Completed. Elapsed Time: {round(end_time - start_time, 2)}s")


def make_year_density_graph(keyword_df, vectorizer, lda_model, n_topics, start_year, end_year, csv_dest_path,
                            graph_dest_path):
    """
    make_year_density graph

    :param keyword_df:
    :param vectorizer:
    :param lda_model:
    :param n_topics:
    :param start_year:
    :param end_year:
    :param dest_path
    """
    start_time = time.time()
    print(f"INFO: Saving Year Desntiy Graph to {graph_dest_path}")

    if platform.system() == "Windows":
        font_name = font_manager.FontProperties(
            fname=str(os.path.join(os.environ["WINDIR"], "Fonts")) + "/malgun.ttf").get_name()
        rc('font', family=font_name)
    elif platform.system() == "Darwin":
        rc("font", family="AppleGothic")

    yearly_data = []
    for i in range(start_year, end_year + 1):
        yearly_data.append(keyword_df[f"{i}"].values)

    yearly_topic_matrices = []
    for year in yearly_data:
        document_term_matrix = vectorizer.transform(year)
        topic_matrix = lda_model.transform(document_term_matrix)
        yearly_topic_matrices.append(topic_matrix)

    yearly_keys = []
    for topic_matrix in yearly_topic_matrices:
        yearly_keys.append(common.get_keys(topic_matrix))

    yearly_counts = []
    for keys in yearly_keys:
        categories, counts = common.keys_to_counts(keys)
        yearly_counts.append(counts)

    yearly_topic_counts = pd.DataFrame(
        data=np.array(yearly_counts), index=range(start_year, end_year + 1))
    yearly_topic_counts.columns = ['Topic {}'.format(i + 1) for i in range(n_topics)]

    yearly_topic_counts.to_csv(csv_dest_path, sep=",", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(14, 10))
    sb.heatmap(yearly_topic_counts, cmap="YlGnBu", ax=ax)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.savefig(graph_dest_path, dpi=300)

    end_time = time.time()
    print(f"INFO: Completed. Elapsed Time: {round(end_time - start_time, 2)}s")
