### chapter 02. 처음해 보는 딥러닝

# 함수 준비하기

from keras.models import Sequential   # 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.layers import Dense

# 모듈 준비하기
import numpy as np                    # 필요한 라이브러리를 불러옵니다.
import tensorflow as tf
if type(tf.contrib) != type(tf):      # warning 출력 안하기
    tf.contrib._warning = None


## 1. 미지의 일을 예측하는 힘



## 2. 폐암 수술 환자의 생존율 예측하기

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다.
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 준비된 수술 환자 데이터를 불러들입니다.
Data_set = np.loadtxt("Dataset/ThoraricSurgery.csv", delimiter=",")

# 환자의 기록과 수술 결과를 X와 Y로 구분하여 저장합니다.
X = Data_set[:,0:17]        # 속성 데이터 셋: 환자의 기록
Y = Data_set[:,17]          # 클래스 데이터 셋: 환자의 결과

# 딥러닝 구조를 결정합니다(모델을 설정하고 실행하는 부분입니다).
model = Sequential()                      # 층층이 쌓는것

model.add(Dense(30,
                input_dim=17,             # 17열을 받아 들이기 위해
                activation='relu'))       # relu라는 모델 사용

model.add(Dense(1,
                activation='sigmoid'))    # sigmoid 모델 사용

# 딥러닝을 실행합니다.
model.compile(loss='mean_squared_error',  #
              optimizer='adam',           #
              metrics=['accuracy'])       #
model.fit(X,                              #
          Y,                              #
          epochs=30,                      #
          batch_size=10)                  #

# 결과를 출력합니다.
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))



## 3. 딥러닝 코드 분석

# 첫 번째 부분: 데이터 분석과 입력

# 두 번째 부문: 딥러닝 실행
#    activation: 다음 층으로 어떻게 값을 넘길지 결정하는 부분입니다.
#    loss: 한 번 신경망이 실행될 때마다 오차 값을 추적하는 함수힙니다.
#    optimizer: 오파를 어떻게 줄여 나갈지 정하는 함수입니다.

# 마지막 부분: 결과 출력