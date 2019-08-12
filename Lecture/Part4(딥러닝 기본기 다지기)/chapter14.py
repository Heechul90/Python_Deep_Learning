### chapter 14. 베스트 모델 만들기

# 함수 준비하기

from keras.models import Sequential  # 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# 모듈 준비하기

import numpy as np                # 필요한 라이브러리를 불러옵니다.
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

if type(tf.contrib) != type(tf):  # warning 출력 안하기
    tf.contrib._warning = None



## 1. 데이터의 확인과 실행
df_pre = pd.read_csv('Dataset/wine.csv', header = None)

# 데이터를 정해진 비율만큼 랜덤으로 뽑아오는 함수
df = df_pre.sample(frac = 1)     # 원본 데이터의 100%를 불러옴
df.info()
df.head()

# 0: 주석산 농도
# 1: 아세트산 농도
# 2: 구연산 농도
# 3: 잔류 당분 농도
# 4: 염화나트륨 농도
# 5: 유리 아황산 농도
# 6: 총 아황산 농도
# 7: 밀도
# 8: pH
# 9: 황산칼륨 농도
# 10: 알코올 도수
# 11: 와인의 맛(0~10등급)
# 12: class(1:레드와인, 0:화이트와인

dataset = df.values
X = dataset[:, :12]
Y = dataset[:, 12]

# 은닉층 4개를 각각 30, 12, 8, 1개의 노드, 이항분류(binary_crossentropy)
# 최적함수: adam, 200회 반복

########## 와인의 종류 예측하기: 데이터 확인과 실행 전테 코드 ################
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 입력
df_pre = pd.read_csv('Dataset/wine.csv', header=None)
df = df_pre.sample(frac=1)

dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]

# 모델 설정
model = Sequential()
model.add(Dense(30,  input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#모델 컴파일
model.compile(loss='binary_crossentropy',
           optimizer='adam',
           metrics=['accuracy'])

# 모델 실행
model.fit(X, Y, epochs=200, batch_size=200)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
#####################################################################



## 2. 모델 업데이트하기
# 100번째 에포크를 실행하고 난 결과 오차가 0.0612라면,
# 파일명은 100-0.0612.hdf5
import os
MODEL_DIR = 'Model'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = 'model/{epch:02d}-{val_loss:.4f}.hdf5'

# 모델을 저장하기 위해 케라스의 콜백 함수중 ModelCheckpoint 함수 부러옴
from keras.callbacks import ModelCheckpoint

# 학습 정확도: acc
# 테스트셋 정확도: val_acc
# 학습셋 오차: loss

checkpointer = ModelCheckpoint(filepath = modelpath,   # 모델 저장될 곳 지정
                               monitor = 'val_loss',   #
                               verbose = 1)            # 1로 정하면 해당함수의 진행 사항 출력, 0은 미출력

# 모델을 학습할 때마다 위에서 정한 checkpointer의 값을 받아 지정된 곳에 모델 저장
model.fit(X, Y,
          validation_split = 0.2,
          epochs = 200,
          batch_size = 200,
          verbose = 0,
          callbacks = [checkpointer])

########## 와인의 종류 예측하기: 모델 업데이트 전체 코드 ################
# 함수, 모듈 준비
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np
import os
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df_pre = pd.read_csv('Dataset/wine.csv', header=None)
df = df_pre.sample(frac=1)

dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]

# 모델의 설정
model = Sequential()
model.add(Dense(30,  input_dim = 12, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 저장 폴더 설정
MODEL_DIR = 'Model/'
if not os.path.exists(MODEL_DIR):
   os.mkdir(MODEL_DIR)

# 모델 저장 조건 설정
modelpath = "Model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath = modelpath,
                               monitor = 'val_loss',
                               verbose = 1,
                               save_best_only = True)

# 모델 실행 및 저장
model.fit(X, Y,
          validation_split = 0.2,
          epochs = 200,
          batch_size = 200,
          verbose = 0,
          callbacks = [checkpointer])



## 3. 그래프로 확인하기
################# 와인의 종류 예측하기: 그래프 표현 전체 코드 ########################
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df_pre = pd.read_csv('Dataset/wine.csv', header=None)
df = df_pre.sample(frac=0.15)

dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]

# 모델의 설정
model = Sequential()
model.add(Dense(30,  input_dim = 12, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 저장 폴더 설정
MODEL_DIR = 'Model/'
if not os.path.exists(MODEL_DIR):
   os.mkdir(MODEL_DIR)

# 모델 저장 조건 설정
modelpath = "Model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath = modelpath,
                               monitor = 'val_loss',
                               verbose = 1,
                               save_best_only = True)

# 모델 실행 및 저장
history = model.fit(X, Y,
                    validation_split = 0.33,
                    epochs = 3500,
                    batch_size = 500)

# y_vloss에 테스트셋으로 실험 결과의 오차 값을 저장
y_vloss = history.history['val_loss']

# y_acc 에 학습 셋으로 측정한 정확도의 값을 저장
y_acc = history.history['acc']

# x값을 지정하고 정확도를 파란색으로, 오차를 빨간색으로 표시
x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vloss, "o", c = "red", markersize = 3)
plt.plot(x_len, y_acc, "o", c = "blue", markersize = 3)

plt.show()
#####################################################################



## 학습의 자동 중단
# 테스트셋 오차가 줄지 않으면 학습을 멈추게 하는 함수: earlyStopping()
from keras.callbacks import EarlyStopping
early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 100)        # 오차가 좋아지지 않아도 몇 번까지 지다릴지

# 이제 앞서 정한대로 에포크 획수와 배치 크기 등을 설정하고
# early_stopping_callback 값을 불러옴
model.fit(X, Y,
          validation_split = 0.33,
          epochs = 3500,
          batch_size = 500,
          callbacks = [early_stopping_callback])

################ 와인의 종류 예측하기: 학습의 자동 중단 전체 코드 ###############
# 함수, 모듈 준비하기
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df_pre = pd.read_csv('Dataset/wine.csv', header = None)
df = df_pre.sample(frac = 0.15)

dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]

model = Sequential()
model.add(Dense(30,  input_dim = 12, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 100)

# 모델 실행
model.fit(X, Y,
          validation_split = 0.2,
          epochs = 2000,
          batch_size = 500,
          callbacks = [early_stopping_callback])

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
###################################################################

##################### 와인의 종류 예측하기: 전체 코드 ###################

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping

import pandas as pd
import numpy as np
import os
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df_pre = pd.read_csv('Dataset/wine.csv', header = None)
df = df_pre.sample(frac = 0.15)

dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]

model = Sequential()
model.add(Dense(30,  input_dim = 12, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 저장 폴더 만들기
MODEL_DIR = 'Model/'
if not os.path.exists(MODEL_DIR):
   os.mkdir(MODEL_DIR)

modelpath = "Model/{epoch:02d}-{val_loss:.4f}.hdf5"

# 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath = modelpath,
                               monitor = 'val_loss',
                               verbose = 1,
                               save_best_only = True)

# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 100)

model.fit(X, Y,
          validation_split = 0.2,
          epochs = 3500,
          batch_size = 500,
          verbose = 0,
          callbacks = [early_stopping_callback, checkpointer])

print('\n Accuracy: %.4f' % (model.evaluate(X, Y)[1]))
####################################################################
