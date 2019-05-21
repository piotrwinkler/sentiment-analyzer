import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer


def use(file):
    """
    This module loads embedding vectors from file and prepares embedding matrix which is used in learning process.
    """
    embedding_dim = 100

    embeddings_index = {}
    words = []
    f = open(file, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        words.append(word)
        vec = np.asarray(values[1:])
        embeddings_index[word] = vec
    f.close()

    words = words[1:]

    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(words)
    word_index = tokenizer_obj.word_index
    print('Found %s unique tokens.' % len(word_index))

    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, tokenizer_obj, num_words
