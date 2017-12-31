import pandas as pd
import numpy as np
import sqlite3
import MeCab

from gensim.models.keyedvectors import KeyedVectors

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential

mc = MeCab.Tagger("-Owakati")

def tokenize(text):
    return mc.parse(text)

def main():

    # Load word2vec model
    w2vmodelpath = 'entity_vector/entity_vector.model.bin'
    word_vectors = KeyedVectors.load_word2vec_format(w2vmodelpath, binary=True)

    # Get embedding layer for keras
    embedding_layer = word_vectors.get_keras_embedding()

    # Load Dataset
    conn = sqlite3.connect("data.db")
    df = pd.read_sql('select * from data', conn)

    # Convert label to score
    score = {'○':1, '□':0, '▲':-1, '×':-2, '◎':2}

    # Get tokenized text list
    print("tokenizing texts...")
    tokenized_text_list = [tokenize(texts) for texts in df.Comment]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokenized_text_list)
    seq = tokenizer.texts_to_sequences(tokenized_text_list)

    # Define training data
    Y = np.array(df.Label.map(score).values).reshape((Y.shape[0],1))
    X = sequence.pad_sequences(seq, maxlen=400)

    # model parameters
    in_out_neurons = 1
    hidden_neurons = 250

    # Define LSTM model
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(hidden_neurons,
                   return_sequences=False,
                   activation='tanh',
                   inner_activation='sigmoid'))
    model.add(Dense(in_out_neurons))
    model.add(Activation("linear"))

    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    # fit model
    model.fit(X, Y, nb_epoch=50)

    # save model
    model.save('LSTM.h5')

    # Close Database
    conn.close()

if __name__ == '__main__':
    main()
