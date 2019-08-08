### chapter 04. 오차 수정하기: 경사 하강법

# 함수 준비하기

from keras.models import Sequential   # 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.layers import Dense

# 모듈 준비하기
import numpy as np                    # 필요한 라이브러리를 불러옵니다.
import tensorflow as tf

if type(tf.contrib) != type(tf):      # warning 출력 안하기
    tf.contrib._warning = None



## 1. 미분의 개념



## 2. 경사 하강법의 개요

# 1) a1에서 미분을 구한다
# 2) 구해진 기울기의 반대방향(기울기가 + 면 음의 방향, -면 양의 방향)으로
#    얼마간 이도시킨 a2에서 미분을 구한다
# 3) a3에서 미분을 구한다
# 4) 3)의 값이 0이 아니면 위 과정을 반복한다.



## 3. 학습률

# 어느 만큼 이동시킬지를 신중히 경정해야 하는데,
# 이때 이동거리를 정해주는 것이 바로 학습률이다.



## 4. 코딩으로 확인하는 경사 하강법

import tensorflow as tf # 머신러닝 및 딥러닝 전문 라이브러리

# x, y의 데이터 값

data = [[2, 81], [4, 93], [6, 91], [8, 97]]

x_data = [x_row[0] for x_row in data]                  # x 값을 리스트로 저장
y_data = [y_row[1] for y_row in data]                  # y 값을 리스트로 저장

# 기울기 a와 y 절편 b의 값을 임의로 정한다.
# 단, 기울기의 범위는 0 ~ 10 사이이며 y 절편은 0 ~ 100 사이에서 변하게 한다.
# tf불러오고 변수의 값을 정할 때는 Variable() 함수 이용
# random_uniform() 임의의 수를 생성

a = tf.Variable(tf.random_uniform([1], 0, 10,          # 0에서 10사이의 임의의 수 1개
                                  dtype = tf.float64,  # 데이터 형식
                                  seed = 0))

b = tf.Variable(tf.random_uniform([1], 0, 100,         # 0에서 100사이의 임의의 수 1개
                                  dtype = tf.float64,  # 데이터 형식
                                  seed = 0))

# y에 대한 일차 방정식 ax+b의 식을 세운다.
y = a * x_data + b

# 텐서플로 RMSE 함수
# tf.sqrt(x): x의 제곱근을 계산
# tf.reduce_mean(x): x의 평균을 계산
# tf.square(x): x의 제곱을 계산

rmse = tf.sqrt(tf.reduce_mean(tf.square( y - y_data ))) # 평균 제곱근 오차의 식

# 학습률 값
learning_rate = 0.1

# RMSE 값을 최소로 하는 값 찾기
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

# 텐서플로를 이용한 학습
with tf.Session() as sess:                       # session 함수를이용해 고동에 필요한 리소스를 컴퓨터에 할당

    sess.run(tf.global_variables_initializer())  # 변수 초기화

    for step in range(2001):                     # 2001번 실행(0번 째를 포함하므로)
        sess.run(gradient_decent)

        if step % 100 == 0:                      # 100번마다 결과 출력
            print("Epoch: %.f, "                 # Epoch: 입력 값에 대해 몇 번이나 반복하여 실험했는지를 타나냄
                  "RMSE = %.04f, "
                  "기울기 a = %.4f, "
                  "y 절편 b = %.4f" % (step,sess.run(rmse),sess.run(a),sess.run(b)))



## 5. 다중 선형 회귀란?

# 독립변수가 2개일 때
# y = a1x1 + a2x2 + b



## 6. 코딩으로 확인하는 다중 선형 회귀

# x1, x2, y의 데이터 값
data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [x_row1[0] for x_row1 in data]
x2 = [x_row2[1] for x_row2 in data]         # 새로 추가되는 값
y_data = [y_row[2] for y_row in data]

# 기울기 a와 y절편 b의 값을 임의로 정함.
# 단 기울기의 범위는 0-10 사이, y 절편은 0-100사이에서 변하게 함
a1 = tf.Variable(tf.random_uniform([1], 0, 10,
                                   dtype=tf.float64,
                                   seed=0))

a2 = tf.Variable(tf.random_uniform([1], 0, 10,
                                   dtype=tf.float64,
                                   seed=0))

b = tf.Variable(tf.random_uniform([1], 0, 100,
                                  dtype=tf.float64,
                                  seed=0))

# 새로운 방정식
y = a1 * x1 + a2 * x2+ b

# 텐서플로 RMSE 함수
rmse = tf.sqrt(tf.reduce_mean(tf.square( y - y_data )))

# 학습률 값
learning_rate = 0.1

# RMSE 값을 최소로 하는 값 찾기
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

# 학습이 진행되는 부분
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(gradient_decent)
        if step % 100 == 0:
            print("Epoch: %.f, RMSE = %.04f, 기울기 a1 = %.4f, 기울기 a2 = %.4f, y절편 b = %.4f" % (step,sess.run(rmse),sess.run(a1),sess.run(a2),sess.run(b)))

