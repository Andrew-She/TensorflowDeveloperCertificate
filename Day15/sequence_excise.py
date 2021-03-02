import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt


# set up
def plot_series(time, series, format="-", start=0, stop=None, label=None):
    plt.plot(time[start:stop], series[start:stop], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonality_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seaonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonality_pattern(season_time)


def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return noise_level * rnd.randn(len(time))


time = np.arange(4 * 365 + 1)

baseline = 10
slope = 0.1
trend_series = baseline + trend(time, slope)
# plot_series(time, trend_series)
# plt.show()

period = 365
amplitude = 40
seasonality_series = seaonality(time, period, amplitude)
# plot_series(time, seasonality_series)
# plt.show()

noise_level = 5
seed = 33
noise_series = white_noise(time, noise_level, seed)
# plot_series(time, noise_series)
# plt.show()

total_series = trend_series + seasonality_series + noise_series
plot_series(time, total_series)
plt.show()

split_time = 1000
time_train = time[:split_time]
x_train = total_series[:split_time]
time_valid = time[split_time:]
x_valid = total_series[split_time:]


def window_dataset(series, window_size=5, batch_size=32):
    dt = tf.data.Dataset.from_tensor_slices(series)
    dt = dt.window(window_size + 1, shift=1, drop_remainder=True)
    dt = dt.flat_map(lambda w: w.batch(window_size + 1))
    dt = dt.map(lambda w: (w[:-1], w[-1]))
    dt = dt.shuffle(buffer_size=len(series))
    return dt.batch(batch_size).prefetch(1)


tf.random.set_seed(33)
np.random.seed(33)
window_size = 30
train_set = window_dataset(x_train, window_size)
valid_set = window_dataset(x_valid, window_size)

epochs = 500
model = tf.keras.models.Sequential()
model.add(layers.Dense(1, input_shape=[window_size]))
optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
loss = tf.keras.losses.Huber()
model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10)
history = model.fit(train_set, epochs=epochs, validation_data=valid_set, verbose=1, callbacks=[early_stopping_cb])

# tf.random.set_seed(33)
# np.random.seed(33)
# window_size = 30
# train_set = window_dataset(x_train, window_size)

# # lr schedule
# epoch = 200
# model = tf.keras.models.Sequential()
# model.add(layers.Dense(1, input_shape=[window_size]))
# lr_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-6 * 10 ** (epoch / 30))
# optimizer = tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9)
# loss = tf.keras.losses.Huber()
# model.compile(optimizer=optimizer, loss=loss, metraics=['mae'])
# history = model.fit(train_set, epochs=epochs, callbacks=[lr_scheduler_cb])
#
# plt.semilogx(history.history['lr'], history.history['loss'])
# plt.axis([1e-6, 1, 0, 20])


def model_forecast(model, series, window_size=5, batch_size=32):
    dt = tf.data.Dataset.from_tensor_slices(series)
    dt = dt.window(window_size, shift=1, drop_remainder=True)
    dt = dt.flat_map(lambda w: w.batch(window_size))
    dt = dt.batch(batch_size).prefetch(1)
    return model.predict(dt)


line_forecast = model_forecast(model, total_series[split_time - window_size:-1], window_size)[:,0]
line_forecast.shape
plot_series(time_valid, x_valid)
#plt.show()
plot_series(time_valid, line_forecast)
plt.show()
tf.keras.metrics.mean_absolute_error(x_valid, line_forecast).numpy()
