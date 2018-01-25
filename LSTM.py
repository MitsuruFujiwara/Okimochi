import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import MeCab

from gensim.models.keyedvectors import KeyedVectors

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential

from sklearn.cross_validation import train_test_split

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
#    conn_val = sqlite3.connect('data20180112.db')

    df = pd.read_sql('select * from data', conn)
#    df_val = pd.read_sql('select * from data', conn_val)

    # drop score = 0 data
    df = df[df['Label'] != '□']
#    df_val = df_val[df_val['Label'] != '□']

    # Convert label to score
    score = {'○':1,'▲':0, '×':0, '◎':1}

    # Get tokenized text list
    print("tokenizing texts...")
    tokenized_text_list = [tokenize(texts) for texts in df.Comment]
#    tokenized_text_list_val = [tokenize(texts) for texts in df_val.Comment]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokenized_text_list)
    seq = tokenizer.texts_to_sequences(tokenized_text_list)
#    seq_val = tokenizer.texts_to_sequences(tokenized_text_list_val)

    # Define training data
    Y = np.array(df.Label.map(score).values)
    Y = np.reshape(Y, (Y.shape[0],1))
#    Y_val = np.array(df_val.Label.map(score).values)
#    Y_val = np.reshape(Y_val, (Y_val.shape[0],1))

    X = sequence.pad_sequences(seq, maxlen=400)
#    X_val = sequence.pad_sequences(seq_val, maxlen=400)

    trX, valX, trY, valY = train_test_split(X, Y, test_size=0.1, random_state=0)

    # model parameters
    in_out_neurons = 1
    hidden_neurons = 200

    # Define LSTM model
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(hidden_neurons,
                   return_sequences=False,
                   activation='tanh',
                   inner_activation='sigmoid',
                   dropout_U=0.3,
                   dropout_W=0.3))
    model.add(Dense(in_out_neurons))
    model.add(Activation('tanh'))

    # Compile model
    history = model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit model
    history = model.fit(trX, trY, nb_epoch=30, validation_data=(valX, valY))

    # save model
    model.save('LSTM.h5')

    # Close Database
    conn.close()

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
