import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def tokenizer_pad_sequence(text, max_num_words = None, max_seq_length = None, tokenizer = None):

    length = [len(x.split(" ")) for x in text]

    if tokenizer == None:
        tokenizer = Tokenizer(num_words=max_num_words)
        tokenizer.fit_on_texts(text)

    sequences = tokenizer.texts_to_sequences(text)

    word_index = tokenizer.word_index

    # MAX_SEQ_LENGTH = np.max(arr_length)
    if max_seq_length == None:
        max_seq_length = int(np.mean(length))

    if max_num_words == None:
        max_num_words = len(word_index)

    # Padding all sequences to same length of `max_seq_length`
    X = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

    return X, max_num_words, max_seq_length, tokenizer
    