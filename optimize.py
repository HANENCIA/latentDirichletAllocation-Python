from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import preprocess_string
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm


def preprocessing(d, stopwords_lst):
    """ tokenizer : 'w1 w2 .... '  =>  ['w1', 'w2', .... ] """
    d = preprocess_string(d)
    return [x for x in d if x not in stopwords_lst]


def make_tokenized_lst(raw_csv_path, col_name, stopwords_csv_path):
    """
    make_tokenized_lst
    
    :param raw_csv_path:
    :param col_name:
    :param stopwords_csv_path:
    """
    start_time = time.time()
    print("INFO: Removing stop words and tokenizing")

    raw_df = pd.read_csv(raw_csv_path)
    stopwords_lst = pd.read_csv(stopwords_csv_path, sep=",", encoding="utf-8")['STOPWORDS'].tolist()

    tokenized_df = raw_df.apply(lambda x: preprocessing(x[col_name], stopwords_lst), axis=1)
    print(f"Num of tokenized dataframe: {len(tokenized_df)}")
    tokenized_lst = tokenized_df.to_list()

    end_time = time.time()
    print(f"INFO: Completed. Elapsed Time: {round(end_time-start_time, 2)}s")

    return tokenized_lst


def get_best_passes(tokenized_lst, dest_dir_path, chunk_size=2000, iteration=600, n_topics=20, max_n_passes=50):
    """
    get_best_passes
    
    :param tokenized_lst:
    :param dest_dir_path:
    :param chunk_size: amount of data for training
    :param iteration: count of loop
    :param n_topics: const topic number (the best calculated)
    :param max_n_passes: 계산할 N passes 최대 값 (5배수로 적용, 4를 입력시 N passes 20까지 계산함)
    """
    start_time = time.time()
    print("INFO: Determining the best # of passes (epochs)")

    os.makedirs(dest_dir_path, exist_ok=True)

    """
    preprocessed_text => word/token list: [[w1, w3, ... w999], [w5, w9, ..., w87], ..., [w1, w10, ..., w100]]
    dictionary => unique token list  =>  idx2word (idx, word):               [uw1, uw2, uw3, uw4, ... uw1000]
    corpus[i]:                                                   [(idx, freq), (idx, freq), ..., (idx, freq)]
    """

    dictionary = corpora.Dictionary(documents=tokenized_lst)
    corpus = [dictionary.doc2bow(document=text) for text in tokenized_lst]

    coherence_scores = []
    perplexity_scores = []
    passes = []

    for i in tqdm(range(max_n_passes + 1)):
        if i == 0:
            p = 1
        else:
            p = i * 5

        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, iterations=iteration, passes=p,
                         alpha='auto', eta='auto', eval_every=1, chunksize=chunk_size)
        passes.append(p)

        cm = CoherenceModel(model=model, corpus=corpus, texts=tokenized_lst, coherence='u_mass')
        coherence = cm.get_coherence()
        perplexity = model.log_perplexity(chunk=corpus)

        coherence_scores.append(coherence)
        perplexity_scores.append(perplexity)

    title_font = {'fontsize': 12, 'fontweight': 'bold'}

    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.plot(passes, coherence_scores)
    plt.title('Coherence distribution with N passes', fontdict=title_font, loc='center', pad=10)
    plt.xticks(np.arange(min(passes), max(passes)+1, step=1))
    plt.xlabel('Number of passes')
    plt.ylabel('Coherence scores')

    plt.savefig(os.path.sep.join([dest_dir_path, f"coherence_npasses_({n_topics}_topics)_lda.png"]), dpi=600)

    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.plot(passes, perplexity_scores)
    plt.title('Perplexity distribution with N passes', fontdict=title_font, loc='center', pad=10)
    plt.xticks(np.arange(min(passes), max(passes)+1, step=1))
    plt.xlabel('Number of passes')
    plt.ylabel('Perplexity scores')
    plt.savefig(os.path.sep.join([dest_dir_path, f"perplexity_npasses_({n_topics}_topics)_lda.png"]), dpi=600)

    result_df = pd.DataFrame(zip(passes, coherence_scores, perplexity_scores),
                             columns=["Passes", "Coherence Scores", "Perplexity Scores"])
    result_df.to_csv(os.path.sep.join([dest_dir_path, f"n_passes_({n_topics}_topics).csv"]), sep=",", index=False,
                     encoding="utf-8-sig")

    end_time = time.time()
    print(f"INFO: Completed. Elapsed Time: {round(end_time-start_time, 2)}s")


def get_best_topics(tokenized_lst, dest_dir_path, chunk_size=2000, iteration=600, n_passes=20, max_n_topics=50):
    """
    get_best_topics

    :param tokenized_lst:
    :param dest_dir_path:
    :param chunk_size:
    :param iteration:
    :param n_passes: best N topics를 결정할 때 기준으로 할 passes의 수
    :param max_n_topics: 계산할 N topics 최대 값 (5배수로 적용, 4를 입력시 N topics 20까지 계산함)
    """
    start_time = time.time()
    print("INFO: Determining the best # of topics")

    os.makedirs(dest_dir_path, exist_ok=True)

    dictionary = corpora.Dictionary(documents=tokenized_lst)
    corpus = [dictionary.doc2bow(document=text) for text in tokenized_lst]

    coherence_val = []
    perplexity_val = []
    n_topics_val = []

    for i in tqdm(range(max_n_topics + 1)):
        if i == 0:
            n_topics = 2
        else:
            n_topics = 5 * i

        n_topics_val.append(n_topics)

        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, iterations=iteration, passes=n_passes,
                       alpha='auto', eta='auto', eval_every=1, chunksize=chunk_size)
        cm = CoherenceModel(model=lda, corpus=corpus, texts=tokenized_lst, coherence='u_mass')

        coherence_ = cm.get_coherence()
        perplexity_ = lda.log_perplexity(chunk=corpus)

        coherence_val.append(coherence_)
        perplexity_val.append(perplexity_)

    title_font = {'fontsize': 12, 'fontweight': 'bold'}

    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.plot(n_topics_val, coherence_val)
    plt.title('Coherence distribution with N topics', fontdict=title_font, loc='center', pad=10)
    plt.xticks(np.arange(min(n_topics_val), max(n_topics_val)+1, step=1))
    plt.xlabel('Number of topics')
    plt.savefig(os.path.sep.join([dest_dir_path, f"coherence_ntopics_({n_passes}_passes)_lda.png"]), dpi=600)

    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.plot(n_topics_val, perplexity_val)
    plt.title('Perplexity distribution with N topics', fontdict=title_font, loc='center', pad=10)
    plt.xticks(np.arange(min(n_topics_val), max(n_topics_val)+1, step=1))
    plt.xlabel('Number of topics')
    plt.ylabel('Perplexity scores')
    plt.savefig(os.path.sep.join([dest_dir_path, f"perplexity_ntopics_({n_passes}_passes)_lda.png"]), dpi=600)

    result_df = pd.DataFrame(zip(n_topics_val, coherence_val, perplexity_val),
                             columns=["N Topics", "Coherence Scores", "Perplexity Scores"])
    result_df.to_csv(os.path.sep.join([dest_dir_path, f"n_topics_({n_passes}_passes).csv"]), sep=",", index=False,
                     encoding="utf-8-sig")

    end_time = time.time()
    print(f"INFO: Completed. Elapsed Time: {round(end_time-start_time, 2)}s")


def main():
    config_json_dir = "./config/"
    config_paths = glob.glob(os.path.sep.join([config_json_dir, "*.json"]))

    for config_path in config_paths:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        raw_csv_path = config['COMMON']['RAW_CSV_PATH']
        stopwords_csv_path = config['COMMON']['STOPWORDS_PATH']

        keyword_col_name = config['COMMON']['KEYWORD_COL_NAME']
        dest_dir_path = config['OPTIMIZE']['RESULT_DIR_PATH']

        os.makedirs(dest_dir_path, exist_ok=True)

        tokenized_lst = make_tokenized_lst(raw_csv_path, keyword_col_name, stopwords_csv_path)

        chunk_size = config['OPTIMIZE']['CHUNK_SIZE']
        iteration = config['OPTIMIZE']['ITERATION']

        # 1. determining the best N passes (epochs)
        n_topics = config['OPTIMIZE']['GET_PASSES_N_TOPICS']
        max_n_passes = config['OPTIMIZE']['GET_PASSES_MAX_N_PASSES']

        get_best_passes(tokenized_lst, dest_dir_path, chunk_size, iteration, n_topics, max_n_passes)

        # 2. determining the best N topics
        n_passes = config['OPTIMIZE']['GET_TOPICS_N_PASSES']
        max_n_topics = config['OPTIMIZE']['GET_PASSES_MAX_N_TOPICS']

        get_best_topics(tokenized_lst, dest_dir_path, chunk_size, iteration, n_passes, max_n_topics)


if __name__ == "__main__":
    main()
