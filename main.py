# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def warning_handler(*args, **kwargs):
    pass


warnings.warn = warning_handler

import glob
import json
import os

import eda
import graph
import lda
import ldavis
import preprocessing
import tsne


def main():
    config_json_dir = "./config/"
    config_paths = glob.glob(os.path.sep.join([config_json_dir, "*.json"]))

    for config_path in config_paths:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        os.makedirs("/".join(config['EDA']['NUM_OF_ARTICLES_GRAPH_PATH'].split("/")[:-1]), exist_ok=True)
        os.makedirs("/".join(config['EDA']['TOP_FREQ_WORDS_GRAPH_PATH'].split("/")[:-1]), exist_ok=True)
        os.makedirs("/".join(config['RESULT_TOPIC_WORDS']['CSV_PATH'].split("/")[:-1]), exist_ok=True)
        os.makedirs("/".join(config['RESULT_TOPIC_WORDS']['GRAPH_PATH'].split("/")[:-1]), exist_ok=True)
        os.makedirs("/".join(config['RESULT_YEAR_TOPIC_DENSITY']['CSV_PATH'].split("/")[:-1]), exist_ok=True)
        os.makedirs("/".join(config['RESULT_YEAR_TOPIC_DENSITY']['GRAPH_PATH'].split("/")[:-1]), exist_ok=True)
        os.makedirs("/".join(config['RESULT_TSNE']['GRAPH_PATH'].split("/")[:-1]), exist_ok=True)
        os.makedirs("/".join(config['RESULT_LDAVIS']['GRAPH_PATH'].split("/")[:-1]), exist_ok=True)
        os.makedirs("/".join(config['DUMP']['MODEL_PATH'].split("/")[:-1]), exist_ok=True)
        os.makedirs("/".join(config['DUMP']['TOPIC_MATRIX_PATH'].split("/")[:-1]), exist_ok=True)

        raw_csv_path = config['COMMON']['RAW_CSV_PATH']
        stopwords_csv_path = config['COMMON']['STOPWORDS_PATH']

        # 1. Make Keyword DataFrame
        date_col_name = config['COMMON']['DATE_COL_NAME']
        keyword_col_name = config['COMMON']['KEYWORD_COL_NAME']
        keyword_df = preprocessing.make_keyword_df(raw_csv_path, date_col_name, keyword_col_name)

        # 2. Remove Stopwords
        keyword_df = preprocessing.remove_stopwords(keyword_df, stopwords_csv_path)

        # 3. EDA: Make bar chart of article counts with year, month, and day
        num_of_articles_graph_path = config['EDA']['NUM_OF_ARTICLES_GRAPH_PATH']
        eda.make_num_of_articles_graph(keyword_df, num_of_articles_graph_path)

        # 4: EDA: Make bar chart of top n words
        top_n = config['EDA']['TOP_FREQ_N']
        top_freq_words_graph_path = config['EDA']['TOP_FREQ_WORDS_GRAPH_PATH']
        eda.make_top_n_words_graph(keyword_df, top_n, top_freq_words_graph_path)

        # 5: Vectorize (TF-IDF Vectorizer)
        max_features = config['VECTORIZER']['MAX_FEATURES']
        tfidf_ngram_max = config['VECTORIZER']['TFIDF_NGRAM_MAX']
        tfidf_max_df = config['VECTORIZER']['TFIDF_MAX_DF']
        tfidf_min_df = config['VECTORIZER']['TFIDF_MIN_DF']

        dtm, vector = preprocessing.tfidf_vectorizer(keyword_df, tfidf_ngram_max, tfidf_max_df, tfidf_min_df,
                                                     max_features)

        # 6: Latent Dirichlet Allocation (LDA)
        n_topics = config['TM']['N_TOPICS']
        num_iteration = config['TM']['N_ITER']
        random_state = config['TM']['RANDOM_STATE']

        lda_model, lda_topic_matrix = lda.topic_modeling(dtm, n_topics, "batch", num_iteration, random_state, -1)

        # 7. Get Results: Keyword with each topic
        n_issue_words = config['RESULT_TOPIC_WORDS']['WORD_COUNT']
        lda_topic_words_path = config['RESULT_TOPIC_WORDS']['CSV_PATH']
        lda.get_topic_words(lda_topic_matrix, dtm, n_issue_words, n_topics, vector, lda_topic_words_path)

        # 8. Get Results: most frequent words bar graph with each topic
        topic_graph_path = config['RESULT_TOPIC_WORDS']['GRAPH_PATH']
        graph.make_topic_graph(lda_topic_matrix, dtm, n_topics, vector, topic_graph_path)

        # 9. Get Results: year-topic density chart
        start_year = config['RESULT_YEAR_TOPIC_DENSITY']['START_YEAR']
        end_year = config['RESULT_YEAR_TOPIC_DENSITY']['END_YEAR']
        year_density_csv_path = config['RESULT_YEAR_TOPIC_DENSITY']['CSV_PATH']
        year_density_graph_path = config['RESULT_YEAR_TOPIC_DENSITY']['GRAPH_PATH']
        graph.make_year_density_graph(keyword_df, vector, lda_model, n_topics, start_year, end_year,
                                      year_density_csv_path, year_density_graph_path)

        # 10: Get Results: T-SNE Graph
        tsne_n_components = config['RESULT_TSNE']['N_COMPONENTS']
        tsne_perplexity = config['RESULT_TSNE']['PERPLEXITY']
        tsne_learning_rate = config['RESULT_TSNE']['LEARNING_RATE']
        tsne_n_iter = config['RESULT_TSNE']['N_ITER']
        tsne_random_state = config['RESULT_TSNE']['RANDOM_STATE']
        tsne_angle = config['RESULT_TSNE']['ANGLE']
        tsne_n_jobs = config['RESULT_TSNE']['N_JOBS']
        tsne_graph_path = config['RESULT_TSNE']['GRAPH_PATH']
        tsne.make_tsne_graph(dtm, lda_topic_matrix, vector, n_topics, tsne_n_components, tsne_perplexity,
                             tsne_learning_rate, tsne_n_iter, tsne_random_state, tsne_angle, tsne_n_jobs,
                             tsne_graph_path)

        # 11: Get Results: LDAvis
        ldavis_graph_path = config['RESULT_LDAVIS']['GRAPH_PATH']
        ldavis.make_ldavis(lda_model, dtm, vector, ldavis_graph_path)

        # 12: Save LDA Model
        lda_model_path = config['DUMP']['MODEL_PATH']
        lda_topic_matrix_path = config['DUMP']['TOPIC_MATRIX_PATH']
        lda.save_lda_model(lda_model, lda_topic_matrix, lda_model_path, lda_topic_matrix_path)

        # 13: Make Topic Distribution
        topic_distribution_top_n = config['TOPIC_DISTRIBUTION']['TOP_N']
        topic_distribution_path = config['TOPIC_DISTRIBUTION']['CSV_PATH']
        lda.get_topic_distribution_top_n(raw_csv_path, lda_model_path, lda_topic_matrix_path, topic_distribution_top_n,
                                         topic_distribution_path)


if __name__ == "__main__":
    main()
