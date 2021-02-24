import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 1000
sentence_size = 120
trunc = 'post'
pad = 'post'
oov = "<OOV>"
train_split = 0.8

sentences = []
labels = []
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

with open(path) as fp:
    lines = csv.reader(fp, delimiter=',')
    next(lines)
    for line in lines:
        labels.append(line[0])
        for w in stopwords:
            sentence = line[1].replace(" "+w+" "," ").replace("  "," ")
        sentences.append(sentence)

train_stcs = sentences[:int(train_split*len(sentences))]
train_labels = labels[:int(train_split*len(labels))]
validation_stcs = sentences[int(train_split*len(sentences)):]
validation_labels = labels[int(train_split*len(labels)):]

tk = Tokenizer(vocab_size, ovv_token=oov)
tk.fit_on_texts(train_stcs)
train_word = tk.index_word

train_seq = tk.texts_to_sequences(train_stcs)
train_pad = pad_sequences(train_seq, maxlen=sentence_size,padding=pad,trunc=trunc)

validation_seq = tk.texts_to_sequences(validation_stcs)
validation_pad = pad_sequences(validation_pad,maxlen=sentence_size,padding=pad,trunc=trunc)

label_tk = Tokenizer()
label_tk.fit_on_texts(train_labels)
label_word_index = label_tk.word_index

train_l_seq = np.array(label_tk.texts_to_sequences(train_labels))
validation_l_seq = np.array(label_tk.texts_to_sequences(validation_labels))

model = tf.keras.Sequential([])
model.add(tf.keras.layers.Embedding(vocab_size, dim,input_length=sentence_size))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(32,activation='relu'))
model.add(tf.keras.layers.Dense(6,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics='accuracy')
model.summary()

num_epochs = 30
history = model.fit(train_pad, train_l_seq, epochs=num_epochs,validation_data=(validation_pad, validation_l_seq))

import matplotlib.pyplot as plt
def  plot_gragh(history, str):
    plt.plot(history.history[str])
    plt.plot(history.history['val_'+str])
    plt.title(str)
    plt.xlabel("Epochs")
    plt.ylabel(str)
    plt.legend([str, 'val_'+str])
    plt.show()
plot_gragh(history, 'acc')
plot_gragh(history, 'loss')