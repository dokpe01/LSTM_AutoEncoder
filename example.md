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
array([[[0.10622289],
        [0.20248923],
        [0.30038676],
        [0.3992009 ],
        [0.49857578],
        [0.5983797 ],
        [0.6986947 ],
        [0.7997707 ],
        [0.901994  ]]], dtype=float32)

- yhat은 입력 데이터 sequence의 feature을 가지고 복원시킨 출력 데이터이며, 입력 데이터와 비슷하게 복원된 결과를 확인할 수 있다.
