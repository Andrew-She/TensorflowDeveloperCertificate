import tensorflow as tf
import string
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

file_of_path = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt')
print(file_of_path)

with open(file_of_path, mode='r') as fp:
    lines = fp.read().lower().split('\n')
#    print(lines)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
total_words = len(tokenizer.word_index) + 1
print(tokenizer.word_index)

seqs = []
for line in lines:
    # print(line)
    seq = tokenizer.texts_to_sequences([line])[0]
    # print(seq)
    for i in range(1, len(seq)):
        seqs.append(seq[:i+1])
    # print(seqs)
max_len = max(len(line) for line in seqs)
padded = np.array(pad_sequences(seqs, maxlen=max_len, padding='pre'))
print(padded[0])
train_data = padded[:, :-1]
train_label = padded[:, -1]
train_labels = tf.keras.utils.to_categorical(train_label, num_classes=total_words)

model = tf.keras.models.Sequential()
model.add(layers.Embedding(total_words, 100, input_length=max_len-1))
model.add(layers.Bidirectional(layers.LSTM(150, return_sequences=True)))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(100))
model.add(layers.Dense(total_words/2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(layers.Dense(total_words, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

epochs = 100
history = model.fit(train_data, train_labels, epochs=epochs, verbose=1)

from matplotlib import pyplot as plt
def plot_gragh(history, str):
    epo = range(0, epochs)
    plt.plot(epo, history.history[str])
    plt.title(str)
    plt.xlabel('EPOCHS')
    plt.ylabel(str)
    plt.show()

plot_gragh(history, 'accuracy')
plot_gragh(history, 'loss')

seed_text = "Help me Obi Wan Kenobi, you're my only hope"
test_len = 100
for _ in range(test_len):
    test_seq = tokenizer.texts_to_sequences([seed_text])[0]
    test_pad = pad_sequences([test_seq], maxlen=max_len-1, padding='pre')
    pre_id = model.predict_classes(test_pad, verbose=0)
    pre_text = ""
    for w, i in tokenizer.word_index.items():
        if i == pre_id:
            pre_text = w
            break
    seed_text += " " + pre_text
print(seed_text)