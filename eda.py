# -*- coding: utf-8 -*-

from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import os
import platform
import time


def make_num_of_articles_graph(kwd_df, dest_path):
    """
    make_num_of_articles graph makes bar chart of article counts with year, month, and day

    :param kwd_df:
    :param dest_path:
    """
    start_time = time.time()
    print(f"INFO: Saving Num of Articles graph to {dest_path}")

    if platform.system() == "Windows":
        font_name = font_manager.FontProperties(
            fname=str(os.path.join(os.environ["WINDIR"], "Fonts")) + "/malgun.ttf").get_name()
        rc('font', family=font_name)
    elif platform.system() == "Darwin":
        rc("font", family="AppleGothic")

    monthly_counts = kwd_df.resample('M').count()
    yearly_counts = kwd_df.resample('A').count()
    daily_counts = kwd_df.resample('D').count()

    fig, ax = plt.subplots(3, figsize=(18, 16))
    ax[0].plot(daily_counts)
    ax[0].set_title('Daily Counts', fontsize=12, fontweight="bold")
    ax[1].plot(monthly_counts)
    ax[1].set_title('Monthly Counts', fontsize=12, fontweight="bold")
    ax[2].plot(yearly_counts)
    ax[2].set_title('Yearly Counts', fontsize=12, fontweight="bold")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.savefig(dest_path, dpi=300)

    end_time = time.time()
    print(f"INFO: Completed. Elapsed Time: {round(end_time - start_time, 2)}s")


def make_top_n_words_graph(kwd_df, top_n, dest_path):
    """
    make_top_n_words_graph makes bar chart of top n words

    :param kwd_df:
    :param top_n:
    :param dest_path:
    """
    start_time = time.time()
    print(f"INFO: Saving Top n Words graph to {dest_path}")

    if platform.system() == "Windows":
        font_name = font_manager.FontProperties(
            fname=str(os.path.join(os.environ["WINDIR"], "Fonts")) + "/malgun.ttf").get_name()
        rc('font', family=font_name)
    elif platform.system() == "Darwin":
        rc("font", family="AppleGothic")

    word_count = Counter(chain(*[str(x).split(" ") for x in kwd_df.tolist()]))
    word_top_n = word_count.most_common(top_n)

    word_top_n_key_lst = [x[0] for x in word_top_n]
    word_top_n_value_lst = [x[1] for x in word_top_n]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(range(len(word_top_n_value_lst)), word_top_n_value_lst)
    ax.set_title('Top Frequent Words', fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(word_top_n_value_lst)))
    ax.set_xticklabels(word_top_n_key_lst, rotation=45)  # rotation='vertical'
    ax.set_xlabel('Word', fontsize=10)
    ax.set_ylabel('Number of occurrence', fontsize=10, fontweight="bold")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.savefig(dest_path, dpi=300)

    end_time = time.time()
    print(f"INFO: Completed. Elapsed Time: {round(end_time - start_time, 2)}s")
