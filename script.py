import os
import sys
import re
import codecs
import argparse
import logging
import shutil
import json
from random import shuffle, randint
from datetime import datetime
from collections import namedtuple, OrderedDict
import multiprocessing
import smart_open
import gensim
import gensim.models.doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import time


def read_lines(path):
    return [line.strip() for line in codecs.open(path, "r", "utf-8")]


def current_time_ms():
    return int(time.time()*1000.0)


def make_timestamped_dir(base_path, algorithm, vector_size, epochs, window):
    suffix = '%s_dim=%d_window=%d_epochs=%d' % (
        algorithm, vector_size, window, epochs)
    output_path = os.path.join(base_path, str(
        current_time_ms())) + '_' + suffix
    clean_make_dir(output_path)
    return output_path


def clean_make_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)


def load_stopwords(stopwords_path):
    logging.info("Loading stopwords: %s", stopwords_path)
    stopwords = read_lines(stopwords_path)
    return dict(map(lambda w: (w.lower(), ''), stopwords))


def run(doc_path, output_base_dir, stopwords_path, vocab_min_count, num_epochs, algorithm, vector_size, alpha, min_alpha, window):
    assert gensim.models.doc2vec.FAST_VERSION > - \
        1, "This will be painfully slow otherwise"

    stopwords = load_stopwords(stopwords_path)

    cores = multiprocessing.cpu_count()

    all_docs = []
    logging.info('Loading documents: %s', doc_path)
    docLabels = []
    docLabels = [f for f in os.listdir(doc_path) if
                 f.endswith(".txt")]
    for doc in docLabels:
        words = open(doc_path+"/" + doc).read().replace("\n",
                                                        " ").replace("।", " ")
        words = re.sub(r'[^\w\s]', " ", words).split()
        words = [w for w in words if w not in stopwords and len(w) > 1]
        tags = [doc]
        all_docs.append(TaggedDocument(words=words, tags=tags))
    negative = 5
    hs = 0

    if algorithm == 'pv_dmc':
        model = Doc2Vec(dm=1, dm_concat=1, vector_size=vector_size, window=window, negative=negative, hs=hs,
                        min_count=vocab_min_count, workers=cores)
    elif algorithm == 'pv_dma':
        model = Doc2Vec(dm=1, dm_mean=1, vector_size=vector_size, window=window, negative=negative, hs=hs,
                        min_count=vocab_min_count, workers=cores)
    elif algorithm == 'pv_dbow':
        model = Doc2Vec(dm=0, vector_size=vector_size, window=window, negative=negative, hs=hs,
                        min_count=vocab_min_count, workers=cores)

    logging.info('Algorithm: %s' % str(model))

    logging.info('Build vocabulary')
    model.build_vocab(all_docs)
    vocab_size = len(model.wv.index_to_key)
    logging.info('Vocabulary size: %d', vocab_size)

    target_dir = make_timestamped_dir(
        output_base_dir, algorithm, model.vector_size, num_epochs, window)
    vocab_path = os.path.join(target_dir, 'vocabulary')
    logging.info('Save vocabulary to: %s', vocab_path)
    with open(vocab_path, 'w') as f:
        term_counts = [[term, value]
                       for term, value in model.wv.key_to_index.items()]
        term_counts.sort(key=lambda x: -x[1])
        for x in term_counts:
            f.write('%s, %d\n' % (x[0], x[1]))

    logging.info('Shuffle documents')
    shuffle(all_docs)

    logging.info('Train model')
    model.train(all_docs, total_examples=len(all_docs),
                epochs=num_epochs, start_alpha=alpha, end_alpha=min_alpha)

    logging.info('Save model to: %s', target_dir)
    model.save(os.path.join(target_dir, 'doc2vec.model'))

    model_meta = {
        'doc_path': doc_path,
        'stopwords_path': stopwords_path,
        'target_dir': target_dir,
        'algorithm': algorithm,
        'window': window,
        'vector_size': vector_size,
        'alpha': alpha,
        'min_alpha': min_alpha,
        'num_epochs': num_epochs,
        'vocab_min_count': vocab_min_count,
        'vocab_size': vocab_size,
        'cores': cores,
        'negative': negative,
        'hs': hs
    }

    model_meta_path = os.path.join(target_dir, 'model.meta')
    logging.info('Save model metadata to: %s', model_meta_path)
    with open(model_meta_path, 'w') as outfile:
        json.dump(model_meta, outfile)


def main():
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s', level=logging.INFO)
    run(doc_path="./docs", output_base_dir="./output", stopwords_path='./hindi_stop.txt', vocab_min_count=10,
        num_epochs=20, algorithm="pv_dmc", vector_size=200, alpha=0.025, min_alpha=0.001, window=5)


if __name__ == '__main__':
    main()
