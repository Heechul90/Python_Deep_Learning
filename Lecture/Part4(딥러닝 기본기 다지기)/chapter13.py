### chapter 13. 과적합 피하기

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
df = pd.read_csv('Dataset/sonar.csv',
                 header = None)
df.info()
df.head()

# seed값 설정하기
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 입력
df = pd.read_csv('Dataset/sonar.csv',
            header = None)
dataset = df.values
X = dataset[:, :60]
Y_obj = dataset[:, 60]

# 문자열 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 모델 설정
model = Sequential()
model.add(Dense(24, input_dim = 60, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일
model.compile(loss = 'mean_squared_error',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 실행
model.fit(X, Y,
          epochs = 200,
          batch_size = 5)

# 결과 출력
print('\n Accuracy: %.4f' % (model.evaluate(X, Y)[1]))



## 2. 과적합 이해하기
# 과적합(over fitting)은 모델이 학습 데이터셋 안세어는 일정 수준 이상의 예측 정확도를 보이지만,
# 새로운 데이터에 적용하면 잘 맞지 않는 것을 말한다



## 3. 학습셋과 테스트셋
# 과적합을 방지하려면?
# 학습을 하는 데이터셋과 이를 테스트할 데이터셋을 완전히 구분한 다음 학습과 동시에 데스트를 병행

# seed값 설정하기
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 입력
df = pd.read_csv('Dataset/sonar.csv',
            header = None)
dataset = df.values
X = dataset[:, :60]
Y_obj = dataset[:, 60]

# 문자열 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 학습셋 70%, 테스트셋 30%로 설정
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size = 0.3,
                                                    random_state = seed)

# 모델 설정
model = Sequential()
model.add(Dense(24, input_dim = 60, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일
model.compile(loss = 'mean_squared_error',
              optimizer = 'adam',
              metrics = ['accuracy'])


# 학습셋으로 학습, 테스트셋으로 테스트
model.fit(X_train, Y_train,
          epochs = 130, batch_size = 5)

# 테스트셋에 모델 적용
print('\n Test Accuracy: %.4f' % (model.evaluate(X_test, Y_test)[1]))



## 4. 모델 저장과 재사용
# 학습이 끝난 후 테스트해 본 결과가 만족스러울 때 이를 모델로 저장하여 새로운 데이터에 사용
from keras.models import load_model

model.save('my_model.h5')

################# 모델을 저장하고 불러오는 내용을 포함한 전체 코드 #################
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('Dataset/sonar.csv', header=None)
'''
print(df.info())
print(df.head())
'''
dataset = df.values
X = dataset[:,0:60]
Y_obj = dataset[:,60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 학습셋과 테스트셋을 나눔
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(24,  input_dim = 60, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'mean_squared_error',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.fit(X_train, Y_train, epochs = 130, batch_size = 5)
model.save('my_model.h5')                                              # 모델을 컴퓨터에 저장

del model                                                              # 테스트를 위해 메모리 내의 모델을 삭제
model = load_model('my_model.h5')                                      # 모델을 새로 불러옴

print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))  # 불러온 모델로 테스트 실행
#######################################################################



## 5. K겹 교차 검증
# 테스트셋의 데이터 양이 적기 때문에 실제로 언마나 잘 작동하는지 확신하기 어려움
# 이러한 단점을 보완하고자 만든 방법이 바로 k겹 교차 검증(k-fold cross validation)

# k겹 교차 검증이란?
# 데이터셋을 여러 개로 나누어 하나씩 테스트셋으로
# 사용하고 나머지를 모두 합해서 학습셋으로 사용하는 방법

# 데이터를 원하는 숫자만큼 쪼개 각각 학습셋과 테스트셋으로 사용
# 10개의 파일로 쪼개 테스트하도록 설정
from sklearn.model_selection import StratifiedKFold

n_fold = 10
skf = StratifiedKFold(n_splits = n_fold, shuffle = True, random_state = 0)

accuracy = []
for train, test in skf.split(X, Y):
    model = Sequential()
    model.add(Dense(24, input_dim = 60, activation = 'rely'))
    model.add(Dense(10, activation = 'rely'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'mean_squared_error',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    model.fit(X[train], Y[train], epochs = 100, batch_size = 5)

    # 정확도(Accuracy) 매번 저장하여 한번에 보여줄수 있게 배열을 만든다
    k_accuracy = '%.4f' % (model.evaluate(X[test], Y[test])[1])
    accuracy.append(k_accuracy)

print('/n %.f fold accuracy:' % n_fold, accuracy)

################# 10-fold cross validation 포함된 전테 코드 #################
# 함수, 모듈 준비
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import numpy
import pandas as pd
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('Dataset/sonar.csv', header=None)

dataset = df.values
X = dataset[:,0:60]
Y_obj = dataset[:,60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 10개의 파일로 쪼갬
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

# 빈 accuracy 배열
accuracy = []

# 모델의 설정, 컴파일, 실행
for train, test in skf.split(X, Y):
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X[train], Y[train], epochs=100, batch_size=5)
    k_accuracy = "%.4f" % (model.evaluate(X[test], Y[test])[1])
    accuracy.append(k_accuracy)

# 결과 출력
print("\n %.f fold accuracy:" % n_fold, accuracy)



