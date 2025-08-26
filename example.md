```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, models, layers, optimizers, utils
```
**Numpy 배열 생성**
```python
sequence = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
print(sequence)
```
array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

**배열 구조 변경**
```python
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))
```

**LSTM-AutoEncoder 모델 생성**
```python
model = models.Sequential()
model.add(layers.LSTM(100, activation='relu', input_shape=(n_in, 1)))
model.add(layers.RepeatVector(n_in))
model.add(layers.LSTM(100, activation='relu', return_sequence=True))
model.add(layers.TimeDistributed(layers.Dense(1)))
model.compile(optimizer='adam', loss='mse')
```

**모델 학습**
```python
model.fit(sequence, sequence, epochs=300, verboss=0)
```

**결과 도출**
```python
yhat = model.predict(sequence)
print(yhat)
```
array([[[0.10622289],<br>
        [0.20248923],<br>
        [0.30038676],<br>
        [0.3992009 ],<br>
        [0.49857578],<br>
        [0.5983797 ],<br>
        [0.6986947 ],<br>
        [0.7997707 ],<br>
        [0.901994  ]]], dtype=float32)

- yhat은 입력 데이터 sequence의 feature을 가지고 복원시킨 출력 데이터이며, 입력 데이터와 비슷하게 복원된 결과를 확인할 수 있다.

<hr/>
**시계열 데이터에 적용**

```python
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
```

```python
master_url_root = "https://raw.githubusercontent.com/numenta/NAB/master/data/"

df_small_noise_url_suffix = "artificialNoAnomaly/art_daily_small_noise.csv"
df_small_noise_url = master_url_root + df_small_noise_url_suffix
df_small_noise = pd.read_csv(
    df_small_noise_url, parse_dates=True, index_col="timestamp"
)

df_daily_jumpsup_url_suffix = "artificialWithAnomaly/art_daily_jumpsup.csv"
df_daily_jumpsup_url = master_url_root + df_daily_jumpsup_url_suffix
df_daily_jumpsup = pd.read_csv(
    df_daily_jumpsup_url, parse_dates=True, index_col="timestamp"
)
```

```python
print(df_small_noise.head())
print(df_daily_jumpsup.head())
```
value<br>
timestamp<br>
2014-04-01 00:00:00  18.324919<br>
2014-04-01 00:05:00  21.970327<br>
2014-04-01 00:10:00  18.624806<br>
2014-04-01 00:15:00  21.953684<br>
2014-04-01 00:20:00  21.909120<br>
                         value<br>
timestamp<br>
2014-04-01 00:00:00  19.761252<br>
2014-04-01 00:05:00  20.500833<br>
2014-04-01 00:10:00  19.961641<br>
2014-04-01 00:15:00  21.490266<br>
2014-04-01 00:20:00  20.187739<br>

```python
fig, ax = plt.subplots()
df_small_noise.plot(legend=False, ax=ax)
plt.show()
```
<img src="https://cdn.discordapp.com/attachments/1043838254694273104/1043841817407393853/image.png?ex=68aee3f2&is=68ad9272&hm=9876a8bfb23b2052b46671437838ede27515a8a144233994fd76147373661c85&" alt="plt.show()" width="70%"/>

```python
fig, ax = plt.subplots()
df_daily_jumpsup.plot(legend=False, ax=ax)
plt.show()
```
<img src="https://cdn.discordapp.com/attachments/1043838254694273104/1043841951033741372/image.png?ex=68aee412&is=68ad9292&hm=43d9689cae33dcf6a1089f16ea058147e97abb50879d3b7f95338f91cc732f50&" alt="plt.show()" width="70%"/>

```python
training_mean = df_small_noise.mean()
training_std = df_small_noise.std()
df_training_value = (df_small_noise - training_mean) / training_std
print("Number of training samples:", len(df_training_value))
```
Number of training samples: 4032

```python
TIME_STEPS = 288

def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


x_train = create_sequences(df_training_value.values)
print("Training input shape: ", x_train.shape)
```
Training input shape:  (3745, 288, 1)

```python
model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()
```
Model: "sequential"<br>
_<br>
 Layer (type)                Output Shape              Param #<br>
=================================================================<br>
 conv1d (Conv1D)             (None, 144, 32)           256<br>

 dropout (Dropout)           (None, 144, 32)           0<br>

 conv1d_1 (Conv1D)           (None, 72, 16)            3600<br>

 conv1d_transpose (Conv1DTra  (None, 144, 16)          1808<br>
 nspose)<br>

 dropout_1 (Dropout)         (None, 144, 16)           0<br>

 conv1d_transpose_1 (Conv1DT  (None, 288, 32)          3616<br>
 ranspose)<br>

 conv1d_transpose2 (Conv1DT  (None, 288, 1)           225<br>
 ranspose)<br>

=================================================================<br>
Total params: 9,505<br>
Trainable params: 9,505<br>
Non-trainable params: 0<br>
__<br>

```python
history = model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)
```

```python
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
```

<img src="https://cdn.discordapp.com/attachments/1043838254694273104/1043842249831759973/image.png?ex=68aee45a&is=68ad92da&hm=be809484e6e9fcaea8e23a75ec2ba76c10abaeefd6055d4b30b90e7e4fda3d19&" alt="plt.show()" width="70%"/>

```python
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.show()

threshold = np.max(train_mae_loss)
print("Reconstruction error threshold: ", threshold)
```

<img src="https://cdn.discordapp.com/attachments/1043838254694273104/1043842343641546842/image.png?ex=68aee470&is=68ad92f0&hm=341608574423ec4a61029ce0c11cd32b445df323b84e76753f66e5ac801032e8&" alt="threshold" width="70%"/>

Reconstruction error threshold:  0.1730786498462495

```python
plt.plot(x_train[0])
plt.plot(x_train_pred[0])
plt.show()
```

<img src="https://cdn.discordapp.com/attachments/1043838254694273104/1043842436125954069/image.png?ex=68aee486&is=68ad9306&hm=d11896d5901463f546112d343f6fb5d4aab7970351bb73b4b7fcd482be4b412c&" alt="plt" width="70%"/>

```python
df_test_value = (df_daily_jumpsup - training_mean) / training_std
fig, ax = plt.subplots()
df_test_value.plot(legend=False, ax=ax)
plt.show()

x_test = create_sequences(df_test_value.values)
print("Test input shape: ", x_test.shape)

x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))
```
<img src="https://cdn.discordapp.com/attachments/1043838254694273104/1043842568263303202/image.png?ex=68aee4a5&is=68ad9325&hm=e3b614dd950bc7a5bb57169b94e1d3db8598c84107c91c24fbc21c6cc3a72f7e&" alt="test" width="70%"/>

Test input shape:  (3745, 288, 1)<br>
118/118 [==============================] - 0s 3ms/step<br>

<img src="https://cdn.discordapp.com/attachments/1043838254694273104/1043842648571658250/image.png?ex=68aee4b9&is=68ad9339&hm=48b1d2efc59c1241a28e97cf4c6ea57b3dca33f9e532e56e89a425a2d8eb71a5&" alt="anomaly sample" width="70%"/>

Number of anomaly samples:  405

```python
anomalous_data_indices = []
for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)
```

```python
df_subset = df_daily_jumpsup.iloc[anomalous_data_indices]
fig, ax = plt.subplots()
df_daily_jumpsup.plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color="r")
plt.show()
```

<img src="https://cdn.discordapp.com/attachments/1043838254694273104/1043842824950525992/image.png?ex=68aee4e3&is=68ad9363&hm=e24d068c5c8b1a90c2ee46c48b0fcbea16eafa1ee5bb3cc1b4c19f3b51c4c969&" alt="anomaly detection" width="70%"/>
