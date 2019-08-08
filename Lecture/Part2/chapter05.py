### chapter 05. 참 거짓 판단 장치: 로지스틱 회귀

# 함수 준비하기

from keras.models import Sequential   # 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.layers import Dense

# 모듈 준비하기
import numpy as np                    # 필요한 라이브러리를 불러옵니다.
import tensorflow as tf

if type(tf.contrib) != type(tf):      # warning 출력 안하기
    tf.contrib._warning = None



## 1. 로지스틱 회귀의 정의
# 직선이 아니라, 참(1)과 거짓(0) 사이를 구분하는 S자 형태의 선을 그어주는 작업



## 2. 시그모이드 함수
# 시그모이드 함수(sigmoid function)는 S자 형태로 그래프가 그려지는 함수
# y = 1 / (1 + np.e ** (-ax + b))

# a값이 작아지면 오차는 무한대로 커지지만,
# a값이 컨진다고 해서 오차가 문한대로 커지지는 않음
# b값은 너무 크거나 작을 경우 오차가 무한대로 커지므로 이차 함수 그래프로 표현



## 3. 오차 공식
# 실제 값이 1이면 오차는 0에 가까움
# 실제 값이 0이면 오차는 1에 가까움
# 로그함수를 이용



## 4. 로그함수
# 예측 값 1 일때 -log h
# 예측 값 0 일때 -log (1 - h)
# -{ylogh + (1 - y)log(1 - h)}



## 5. 코딩으로 확인하는 로지스틱 회귀

# x,y의 데이터 값
data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

# a와 b의 값을 임의로 정함
a = tf.Variable(tf.random_normal([1],
                                 dtype=tf.float64,
                                 seed=0))

b = tf.Variable(tf.random_normal([1],
                                 dtype=tf.float64,
                                 seed=0))

# y 시그모이드 함수의 방정식을 세움
y = 1 / (1 + np.e ** (a * x_data + b))

# loss를 구하는 함수
loss = -tf.reduce_mean(np.array(y_data) * tf.log(y) + (1 - np.array(y_data)) * tf.log(1 - y))

# 학습률 값
learning_rate = 0.5

# loss를 최소로 하는 값 찾기
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(60001):
        sess.run(gradient_decent)
        if i % 6000 == 0:
            print("Epoch: %.f, loss = %.4f, 기울기 a = %.4f, 바이어스 b = %.4f" % (i, sess.run(loss), sess.run(a), sess.run(b)))



## 6. 여러 입력 값을 갖는 로지스틱 회귀

# 실행할 때마다 같은 결과를 출력하기 위한 seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# x,y의 데이터 값
x_data = np.array([[2, 3],[4, 3],[6, 4],[8, 6],[10, 7],[12, 8],[14, 9]])
y_data = np.array([0, 0, 0, 1, 1, 1,1]).reshape(7, 1)

# 입력 값을 플래이스 홀더에 저장
X = tf.placeholder(tf.float64, shape=[None, 2])
Y = tf.placeholder(tf.float64, shape=[None, 1])

# 기울기 a와 bias b의 값을 임의로 정함.
a = tf.Variable(tf.random_uniform([2, 1],
                                  dtype=tf.float64))   # [2,1] 의미: 들어오는 값은 2개, 나가는 값은 1개
b = tf.Variable(tf.random_uniform([1],
                                  dtype=tf.float64))

# y 시그모이드 함수의 방정식을 세움
y = tf.sigmoid(tf.matmul(X, a) + b)

# 오차를 구하는 함수
loss = -tf.reduce_mean(Y * tf.log(y) + (1 - Y) * tf.log(1 - y))

# 학습률 값
learning_rate = 0.1

# 오차를 최소로 하는 값 찾기
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

predicted = tf.cast(y > 0.5, dtype=tf.float64)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float64))

# 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3001):
        a_, b_, loss_, _ = sess.run([a, b, loss, gradient_decent], feed_dict={X: x_data, Y: y_data})
        if (i + 1) % 300 == 0:
            print("step=%d, a1=%.4f, a2=%.4f, b=%.4f, loss=%.4f" % (i + 1, a_[0], a_[1], b_, loss_))


# 어떻게 활용하는가
    new_x = np.array([7, 6.]).reshape(1, 2)  #[7, 6]은 각각 공부 시간과 과외 수업수.
    new_y = sess.run(y, feed_dict={X: new_x})

    print("공부 시간: %d, 개인 과외 수: %d" % (new_x[:,0], new_x[:,1]))
    print("합격 가능성: %6.2f %%" % (new_y*100))



# 7. 실제 값 적용하기
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

# 어떻게 활용하는가
    new_x = np.array([7, 6.]).reshape(1, 2)  #[7, 6]은 각각 공부 시간과 과외 수업수.
    new_y = sess.run(y, feed_dict={X: new_x})

    print("공부 시간: %d, 개인 과외 수: %d" % (new_x[:,0], new_x[:,1]))
    print("합격 가능성: %6.2f %%" % (new_y*100))



## 8. 로지스틱 회귀에서 퍼셉트론으로
