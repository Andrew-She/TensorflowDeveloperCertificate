import sys
import tensorflow as tf
from tensorflow.keras import layers as layers

import numpy as np
from matplotlib import pyplot as plt

#get data from datasets
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, y_train, x_validation, y_validation = x_train[:50000],  y_train[:50000], x_train[50000:],  y_train[50000:]
x_train, x_validation, x_test = x_train/255., x_validation/255., x_test/255.
print(len(y_train),len(y_test))
#preprocessing data

#callback
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_callback_model.h5",
                                                   save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                    restore_best_weights=True,)
#create model
model = tf.keras.Sequential()
model.add(layers.Flatten(input_shape=(28,28)))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
#compile model
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#fit model
EPOCHS = 30
history = model.fit(x_train,y_train,epochs=EPOCHS,
                    shuffle=True,
                    validation_data=(x_validation,y_validation),
                    callbacks=[checkpoint_cb, early_stopping_cb])
#evaluate & predict
model.evaluate(x_test, y_test)
x_prob = x_test[:3]
y_prob = model.predict(x_prob).round(2)
print(y_prob)

x_prob_class = x_test[:5]
x_prob_class = np.argmax(model.predict(x_prob_class), axis=-1)
print(x_prob_class)
#save & restore
model.save("my_fashin_mnist_model.h5")
pre_model = tf.keras.models.load_model("my_callback_model.h5")
pre_model.evaluate(x_test, y_test)
print(pre_model)
#plot loss and acc
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.plot(epochs_range, acc, label= 'Epochs')
plt.plot(epochs_range, val_acc, label = 'Accuracy')
plt.title("ACCURACY")
plt.grid(True)
plt.legend(['acc','val_acc'])
plt.figure()
plt.show()

plt.plot(epochs_range, loss, label= 'Epochs')
plt.plot(epochs_range, val_loss, label = 'Loss')
plt.title("LOSS")
plt.grid(True)
plt.legend(['loss','val_loss'])
plt.figure()
plt.show()