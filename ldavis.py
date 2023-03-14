# -*- coding: utf-8 -*-

import pyLDAvis.sklearn
import time


def make_ldavis(lda_model, keyword_dtm, vectorizer, dest_path):
    """
    make_ldavis makes LDAvis html file

    :param lda_model:
    :param keyword_dtm:
    :param vectorizer:
    :param dest_path
    """
    start_time = time.time()
    print("INFO: Saving LDAvis to " + str(dest_path))

    lda_vis = pyLDAvis.sklearn.prepare(lda_model, keyword_dtm, vectorizer)
    # pyLDAvis.show(lda_vis)
    pyLDAvis.save_html(lda_vis, dest_path)

    end_time = time.time()
    print("INFO: completed! elapsed time " + str(round(end_time - start_time, 2)) + "s")