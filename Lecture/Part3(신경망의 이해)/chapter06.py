### chapter 06. 퍼셉트론

# 함수 준비하기

from keras.models import Sequential  # 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.layers import Dense

# 모듈 준비하기
import numpy as np  # 필요한 라이브러리를 불러옵니다.
import tensorflow as tf

if type(tf.contrib) != type(tf):  # warning 출력 안하기
    tf.contrib._warning = None

## 1. 가중치, 가중합, 바이어스, 활성화 함수
# y = wx + b (w(weight): 가중치, b(bias): 바이어스



## 2. 퍼셉트론의 과제
# 경우에 따라서 선을 아무리 그어도 해결되지 않는 상황이 있다



## 3. XOR(exclusive OR) 문제
# 이를 해결한 개념이 바로 다층 퍼셉트론(multilayer perceptron)

