### chapter 16. 이미지 인식의 꽃, CNN 익히기

# 함수 준비하기

#-*- coding: utf-8 -*-

from keras.models import Sequential  # 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint,EarlyStopping

# 모듈 준비하기

import numpy as np                # 필요한 라이브러리를 불러옵니다.
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os

if type(tf.contrib) != type(tf):  # warning 출력 안하기
    tf.contrib._warning = None



## 1. 데이터 전처리
# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# MNIST데이터셋 불러오기
(X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data()
print('학습셋 이미지 수: %d 개' % (X_train.shape[0]))
print('테스트셋 이미지 수: %d 개' % (X_test.shape[0]))

# 그래프로 확인
plt.imshow(X_train[0], cmap = 'Greys')
plt.show()

# 코드로 확인
for x in X_train[0]:
    for i in x:
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')

# 28X28의 2차원 배열을 784개의 1차원 배열로 바꿔줌(차원 변환 과정)
X_train = X_train.reshape(X_train.shape[0], 784)
X_train

# 0과 1로 값을 바꿔주기 위해 255로 나눠준다
X_train = X_train.astype('float64')
X_train = X_train / 255
X_train

# X_test도 같이 적용함
X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255
X_test

# 클래스 값 확인
print("class : %d " % (Y_class_train[0]))

# 바이너리화 과정
Y_train = np_utils.to_categorical(Y_class_train, 10)
Y_test = np_utils.to_categorical(Y_class_test, 10)

# 변환된 값 출력
print(Y_train[0])



## 2. 딥러닝 기본 프레임 만들기
# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# MNIST 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

# 모델 프레임 설정
# 은닉층 노드 512, 속성 784, 함수 relu
# 출력층 노드 10, 함수 softmax
model = Sequential()
model.add(Dense(512, input_dim = 784, activation = 'relu'))   # 속성이 784개
model.add(Dense(10, activation = 'softmax'))                  # class가 10개

# 모델 실행 환경 설정
model.compile(loss = 'categorical_crossentropy',              # 다중 분류 문제
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 최적화 설정
MODEL_DIR = 'Lecture/Part5(딥러닝의 활용)/Model1/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "Lecture/Part5(딥러닝의 활용)/Model1/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath = modelpath,
                               monitor = 'val_loss',
                               verbose = 1,
                               save_best_only = True)
early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 10)

# 모델의 실행
history = model.fit(X_train, Y_train,
                    validation_data = (X_test, Y_test),
                    epochs = 30,
                    batch_size = 200,
                    verbose = 0,
                    callbacks = [early_stopping_callback,checkpointer])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker = '.', c = "red", label = 'Testset_loss')
plt.plot(x_len, y_loss, marker = '.', c = "blue", label = 'Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc = 'upper right')
plt.axis([0, 20, 0, 0.35])
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()



## 3. 더 깊은 딥러닝
# 위에서 했던 딥러닝 프레임은 하나의 은닉층을 둔 아주 단순한 모델
# 딥러닝은 이러한 기본 모델을 바탕으로,
# 프로젝트에 맞춰서 어떤 옵션을 더하고 어떤 층을 추가하느냐에 따라 성능이 더 좋아짐



## 4. 컴볼루션 신경망(CNN)
model.add(Conv2D(32,                        # 마스크를 32개 적용(필터라고도 함)
                 kernel_size = (3, 3),      # 커널의 크기를 정함 (행 X 렬)
                 input_shape = (28, 28, 1), # 맨 처음 층에는 입력되는 값을 알려줌(행 , 렬, 색상3 또는 흑백1)
                 activation = 'relu'))      # 활성화 함수 정의

# 컨볼루션 층을 하나더 추가
model.add(Conv2D(64, (3, 3), activation = 'relu'))




## 5. 맥스 풀링
# 보통 Max값으로 축소함
model.add(MaxPooling2D(pool_size = 2))

# 드롭아웃(drop out), 플래튼(flatten)
model.add(Dropout(0.25))  # 25%의 노드를 끄고 싶을 때
model.add(Flatten())



## 6. 컨볼루션 신경망 실행하기

############### MNIST 손글씨 인식하기: 컴볼루션 신경망 적용 ###############

# 함수 준비하기
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 불러오기

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# 컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,  activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 최적화 설정
MODEL_DIR = 'Lecture/Part5(딥러닝의 활용)/model2/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath="Lecture/Part5(딥러닝의 활용)/model2/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath = modelpath,
                               monitor = 'val_loss',
                               verbose = 1,
                               save_best_only = True)
early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 10)

# 모델의 실행
history = model.fit(X_train, Y_train,
                    validation_data = (X_test, Y_test),
                    epochs = 30,
                    batch_size = 200,
                    verbose = 0,
                    callbacks = [early_stopping_callback,checkpointer])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker = '.', c = "red", label = 'Testset_loss')
plt.plot(x_len, y_loss, marker = '.', c = "blue", label = 'Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
