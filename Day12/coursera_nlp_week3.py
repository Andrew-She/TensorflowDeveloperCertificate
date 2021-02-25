import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

embedding_dims = 100
max_lengh = 16
trunc_type = 'post'
padding_type = 'post'
oov = "<OOV>"
train_split = 0.9

sentences = []
labels = []
with open(path) as fp:
    lines = csv.reader(fp, delimiter=',')
    #next(lines)
    for line in lines:
        if '0' == line[0]:
            labels.append(line[0])
        else:
            labels.append(line[0])
        sentences.append(line[5])

tk = Tokenizer(oov_token=oov)
tk.fit_on_texts(sentences)

word_index = tk.word_index
vocab_size = len(word_index)

seq = tk.texts_to_sequences(sentences)
padded = pad_sequences(seq, max_lengh=max_lengh,padding=padding_type,truncating=trunc_type)

split = int(train_split*len(labels))
train_sentences = padded[:split]
train_labels = labels[:split]
test_sentences = padded[split:]
test_labels = labels[split:]

embedding_index = {}
with open(path) as fp2:
    for line in fp2:
        sep = line.spit()
        embedding_index[sep[0]] = np.asarray(sep[1:], dtype='float32')
embedding_matrix = np.zeros((vocab_size+1, embedding_dims))
for w, i in word_index.items():
    embedding_vector = embedding_index.get(w)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = tf.keras.Sequential([])
model.add(layers.Embedding(vocab_size,embedding_dims,input_length=max_lengh,weights=[embedding_matrix],trainable=False))
model.add(layers.Dropout(0.2))
model.add(layers.Conv1D(64, 5, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=4))
model.add(layers.LSTM(64))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

epochs = 30
train_sentences = np.array(train_sentences)
train_labels = np.array(train_labels)
test_sentences = np.array(test_sentences)
test_labels = np.array(test_labels)

history = model.fit(train_sentences, train_labels, epochs=epochs, validation_data=(test_sentences, test_labels), verbose=2)

import matplotlib.pyplot as plt
def plot_gragh(history, str):
    plt.plot(str)
    plt.plot('val_'+str)
    plt.tile(str)
    plt.xlabel('Epochs')
    plt.ylabel(str)
    plt.legend([str,'val_'+str])
    plt.show()

plot_gragh(history, 'loss')
plot_gragh(history, 'accuracy')