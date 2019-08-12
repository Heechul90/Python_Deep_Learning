### 과제 26

# 함수 준비하기

from keras.models import Sequential  # 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

# 모듈 준비하기

import numpy as np                # 필요한 라이브러리를 불러옵니다.
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

if type(tf.contrib) != type(tf):  # warning 출력 안하기
    tf.contrib._warning = None

# 한글 사용하기
import platform
from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')




# 데이터 불러오기
df_pre = pd.read_csv('Homework/wine.csv', header = None)
df = df_pre.sample(frac = 1)
df.head()
df.info()

dataset = df.values
X = dataset[:, :12]
Y = dataset[:, 12]

# Y값을 0,1로 바꾸기
from keras.utils import np_utils
Y_encoded = np_utils.to_categorical(Y, 11)

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 모델 설정
# 30, input=12, relu
# 24, relu
# 12, relu
# node= 11, softmax

model = Sequential()
model.add(Dense(30, input_dim = 12, activation = 'relu'))
model.add(Dense(24, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(11, activation = 'softmax'))

# 모델 컴파일
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 저장 폴더 만들기
MODEL_DIR = 'Homework/Model2/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
modelpath = 'Homework/Model2/{epoch:02d}-{val_loss:.4f}.hdf5'

# 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath = modelpath,
                               monitor = 'val_loss',
                               verbose = 1,
                               save_best_only = True)

##################### Best model ##########################
# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 100)

# 모델 실행
model.fit(X, Y_encoded,
          validation_split = 0.3,
          epochs = 2000,
          batch_size = 500,
          callbacks = [early_stopping_callback, checkpointer])

# 결과 출력
print('\n Accuracy: %.4f' % (model.evaluate(X, Y_encoded)[1]))

########################### 그래프 ############################
# 모델 실행 및 저장
history = model.fit(X, Y_encoded,
                    validation_split = 0.3,
                    epochs = 2000,
                    batch_size = 500)

# y_vloss 에 테스트셋으로 실험 결과의 오차 값을 저장
y_vloss = history.history['val_loss']

# y_acc 에 학습셋으로 측정한 정확도의 값을 저장
y_acc = history.history['acc']

# x값을 지정하고 정확도를 파란색으로, 오차를 빨간색으로 표시
x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vloss, 'o', c = 'red', markersize = 3)
plt.plot(x_len, y_acc, 'o', c = 'blue', markersize = 3)
plt.xlabel('에포크')
plt.ylabel('정확도')
plt.title('Model2')
plt.show()





