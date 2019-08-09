### chapter 10. 모델 설계하기

# 함수 준비하기

from keras.models import Sequential  # 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.layers import Dense

# 모듈 준비하기

import numpy as np                # 필요한 라이브러리를 불러옵니다.
import tensorflow as tf

if type(tf.contrib) != type(tf):  # warning 출력 안하기
    tf.contrib._warning = None



## 1. 모델의 정의
# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다.
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 준비된 수술 환자 데이터를 불러들입니다.
Data_set = np.loadtxt("Dataset/ThoraricSurgery.csv", delimiter = ",")

# 환자의 기록과 수술 결과를 X와 Y로 구분하여 저장합니다.
X = Data_set[:,0:17]
Y = Data_set[:,17]

# 딥러닝 구조를 결정합니다(모델을 설정하고 실행하는 부분입니다).
model = Sequential()
model.add(Dense(30, input_dim = 17, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 딥러닝을 실행합니다.(정해진 모델을 컴퓨터가 알아들을 수 있게끔 컴파일 하는 부분)
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=10)

# 결과를 출력합니다.
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))



## 2. 입력층, 은닉층, 출력층
# 딥러닝의 구조를 짜고 층을 설정하는 부분
# 입력층
model = Sequential()

# 은닉층: 데이터에서 17개의 값을 받아 은닉층의 30개 노드로 보낸다
model.add(Dense(30,                          # 30개의 노드를 만들라
                input_dim = 17,              # 입력데이터로부터 몇 개의 값이 들어올지를 정함
                activation = 'relu'))        #

# 출력층
model.add(Dense(1,
                activation = 'sigmoid'))



## 3. 모델 컴파일
model.compile(loss = 'mean_squared_error',   # 오차 함수를 정함
              optimizer = 'adam',            # 최적화의 함수를 선택
              metrics = ['accuracy'])        # 모델 수행 결과를 나타내게끔 설정



## 4. 교차 엔트로피
# 평균 제곱 계열
# mean_squared_error              평균 제곱 오차
#   계산: mean(square(yt - y0))
# mean_absolute_error             평균 절대 오차(실제 값과 예측 값 차이의 절댓값 평균)
#   계산: mean(abs(yt - y0))
# mean_absolute_percentage_error  평균 절대 백분율 오차(절댓값 오차를 절댓값으로 나눈 후 평균)
#   계산: mean(abs(yt - y0) / abs(yt))(단, 분모 != 0)
# mean_squared_logarithmic_error  평균 제곱 로그 오차(실제 값과 예측 값에 로그를 적용한 값의 차이를 제곱한 값의 평균
#   계산: mean(square((log(y0) + 1) - (log(yt) + 1)))

# 교차 엔트로피 계열
# categorical_crossentropy        범주형 교차 엔트로피(일반적인 분류)
# binary_crossentropy             이항 교차 엔츠로피(두 개의 클래스 중에서 예측할 때)



## 5. 모델 실행하기
# 모델을 정의하고 컴파일하고 나면 이제 실해키실 차례
model.fit(X, Y,
          epochs = 30,      # 프로세스가 모든 샘플에 대해 30번 실행
          batch_size = 10)  # 샘플(470개)을 한 번에 10개씩 처리