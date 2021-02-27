import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

data = "In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."

lines = data.split('\n')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
total_len = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(lines)
print(sequences)

max_len = max(len(sequence) for sequence in sequences)
seq_new = []
for seq in sequences:
    if len(seq) >= 2:
        for i in range(2, len(seq)+1):
            seq_new.append(seq[:i])
print(seq_new)

padded = np.array(pad_sequences(seq_new, maxlen= max_len, padding='pre', truncating='post'))
print(padded)
train_data = padded[:, :-1]
train_label = padded[:, -1]
print(train_data[0], train_label[0])
train_labels = tf.keras.utils.to_categorical(train_label, num_classes=total_len)
print(train_data[6], train_labels[6])

model = tf.keras.models.Sequential()
model.add(layers.Embedding(total_len, 64, input_length=max_len-1))
model.add(layers.Bidirectional(layers.LSTM(20)))
model.add(layers.Dense(total_len, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 500
history = model.fit(train_data, train_labels, epochs=epochs)

import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss = history.history['loss']
plt.plot(acc)
plt.plot(loss)
plt.title("ACC&LOSS")
plt.xlabel('EPOCHS')
plt.ylabel('acc&loss')
plt.legend(['ACC', 'LOSS'])
plt.show()

test_text = "Laurence went to dublin"
test_len = 100

for i in range(test_len):
    test_tk = tokenizer.texts_to_sequences(test_text)
    test_pad = pad_sequences(test_tk, maxlen=max_len-1, padding='pre')
    predict = model.predict_classes(test_pad)
    output_word = ""
    for w, index in tokenizer.word_index.items():
        if index == predict:
            output_word = w
        else:
            continue
        test_text += " " + output_word
print(test_text)
