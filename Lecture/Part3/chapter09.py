### chapter 09. 신경망에서 딥러닝으로

# 함수 준비하기

from keras.models import Sequential  # 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.layers import Dense

# 모듈 준비하기
import numpy as np  # 필요한 라이브러리를 불러옵니다.
import tensorflow as tf

if type(tf.contrib) != type(tf):  # warning 출력 안하기
    tf.contrib._warning = None



## 1. 기울기 소실(vanishing gradient) 문제와 활성화 함수
# 렐루(ReLU)함수가 현재 가장 많이 사용되는 활성화 함수
# x가 0보다 작을 때는 모든 값을 0으로 처리,
# 0보다 큰 값은 x를 그대로 사용하는 방법



## 2. 속도와 정확도 문제를 해결하는 고급 경사 하강법
# 확률적 경사 하강법(Stochastic Gradient Descent, SGD)
# 확률적 경사 하강법은 랜덤으로 추출한 데이터를 사용

# 모멘텀(momentum)
# 오차를 수정하기 전 바로 앞 수정 값과 방향(+, -)을 참고하여
# 같은 방향으로 일정한 비율만 수정되게 하는 방법

# 고급 경사 하강법
# 1) 확률적 경사 하강법(SGD)
# 랜덤하게 추출한 일부 데이터를 사용해 더 빨리, 자주 업데이트를 하게 하는것
keras.optimizers.SGD(lr = 0.1)  # 케라스 최적화 함수를 이용

# 2) 모멘텀(Momentum)
# 관성의 방향을 고려해 진동과 폭을 줄이는 효과
keras.optimizers.SGD(lr = 0.1,
                     momentum = 0.9) # 모멘텀 계수를 추가

# 3) 네스테로프 모멘텀(NAG)
# 모멘텀이 이동시킬 방향으로 미리 이동해서 그레이디언트를 계산.
# 불필요한 이동을 줄이는 효과
keras.optimizers.SGD(lr = 0.1,
                     momentum = 0.9,
                     nesterov = True) # 네스테로프 옵션을 추가

# 4) 아다그라드(Adagrad)
# 변수의 업데이트가 잦으면 학습률을 적게 하여 이동 보폭을 조절하는 방법
keras.optimizers.Adagrad(lr = 0.01,         # 아다그라드 함수를 이용
                         epsilon = 1e - 6)
# 참고: epsilon, rho, decay 같은 파라미터는 바꾸지 않고 그대로 사용하기를 권장
#      따라서, learning rate(학습률)값만 적절히 조절

# 5) 알엠에스프롭(RMSProp)
# 아다그라드의 보폭 민감도를 보완한 방법
keras.optimizers.RMSprop(lr = 0.001,   # 알엠에스프롭 함수를 이용
                         rho = 0.9,
                         epsilon = 1e - 8,
                         decay = 0.0)

# 6) 아담(Adam)
# 모멘텀과 알엠에스프롭 방법을 합친 방법
keras.optimizers.Adam(lr = 0.001,
                      beta_1 = 0.9,
                      rho = 0.9,
                      epsilon = 1e - 8,
                      decay = 0.0))