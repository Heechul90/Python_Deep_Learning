### chapter 15. 선형 회귀 적용하기

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



## 1. 데이터 확인하기
df = pd.read_csv('Dataset/housing.csv', delim_whitespace = True, header = None)
df.info()
df.head()

# 속성과 클래스
# 0 - CRIM:     인구 1인당 범죄 발생 수
# 1 - ZN:       25,000평방 피트 이상의 주거 구역 비중
# 2 - INDS:     소매업 외 상업이 차지하는 면적 비율
# 3 - CHAS      찰스강 위치 변수(1: 강 주변, 0: 이외)
# 4 - NOX:      일산화질소 농도
# 5 - RM:       집의 평균 방 수
# 6 - AGE:      1940년 이전에 지어진 비율
# 7 - DIS:      5가지 보스턴 시 고용 시설까지의거리
# 8 - RAD:      순환고속도로의 접근 용이성
# 9 - TAX:      %10,000당 부동산 세율 총계
# 10 - PTRATIO: 지역별 학생과 교사 비율
# 11 - B:       지역별 흑인 비율
# 12 - LSTAT:   급여가 낮은 직업에 종사하는 인구 비율(%)
# 13 - 가격(단위: $1,000)



## 2. 선형 회귀 실행
# 선형 회귀 데이터는 참과 거짓을 구분할 필요 없음
# 출력층에 활성화 함수를 지정할 필요 없음
model = Sequential()
model.add(Dense(30, input_dim = 13, activation = 'relu'))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(1))

# 모델의 학습이 어느 정도 되었는지 확인하기 위해 예측 값과 실제 값을 비교하는 부분을 추가
Y_prediction = model.predict(X_test).flatten()     # 데이터 배열이 몇 차원이든 1차원으로 바꿔 읽기 쉽게 해줌
for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print('실제가격: {:.3f}, 예상가격: {:.3f}'.format(label, prediction))



########################## 보스턴 집값 예측하기 ##################################
#-*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 불러오기
df = pd.read_csv("Dataset/housing.csv", delim_whitespace = True, header = None)

# 데이터 확인하기
'''
print(df.info())
print(df.head())
'''

dataset = df.values
X = dataset[:,0:13]
Y = dataset[:,13]

# 학습셋 70%, 테스트셋 30% 설정
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size = 0.3,     # 테스트셋 30%
                                                    random_state = seed)

# 모델 설정하기
model = Sequential()
model.add(Dense(30, input_dim = 13, activation = 'relu'))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(1))

# 모델 컴파일 하기
model.compile(loss = 'mean_squared_error',
              optimizer = 'adam')

# 모델 실행
model.fit(X_train, Y_train, epochs = 200, batch_size = 10)

# 예측 값과 실제 값의 비교
Y_prediction = model.predict(X_test).flatten()
for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label, prediction))

