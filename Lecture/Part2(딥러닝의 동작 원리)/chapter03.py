### chapter 03. 가장 훌륭한 예측선 긋기: 선형 회귀

# 함수 준비하기

from keras.models import Sequential   # 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.layers import Dense

# 모듈 준비하기
import numpy as np                    # 필요한 라이브러리를 불러옵니다.
import tensorflow as tf

if type(tf.contrib) != type(tf):      # warning 출력 안하기
    tf.contrib._warning = None


## 1. 선형 회귀의 정의



## 2. 가장 훌륭한 예측선이란?



## 3. 최소 제곱법



## 4. 코딩으로 확인하는 최소 제곱

# 독립 변수와 종속 변수 입력
x = [2, 4, 6, 8]            # 학생들의 공부한 시간
y = [81, 93, 91, 97]        # 학생들의 점수

# 평균 구하기
mx = np.mean(x)             # 학생들의 공부한 시간 평균
my = np.mean(y)             # 학생들의 점수 평균

print('x의 평균값:', mx)
print('y의 평균값:', my)

# x 증가량의 제곱: x의 평균값과 x의 각 원소들의 차를 제곱
divisor = sum([(mx - i) ** 2 for i in x])

# y 증가량
def top(x, mx, y, my):
    d = 0                                 # d의 초긱값을 0으로 설정
    for i in range(len(x)):               # x의 개수만큼 실행
        d += (x[i] - mx) * (y[i] - my)    # x의 각 원소와 평균의 차, y의 각 원소와 평균의 차를 곱해서 더하기
    return d

dividend = top(x, mx, y, my)

print('분모:', divisor)
print('분자:', dividend)

# 기울기 구하기
a = dividend / divisor

# y 절편 구하기
b = np.mean(y) - (np.mean(x) * a)
b = my - (mx * a)

print('기울기:', a)
print('y 절편:', b)



## 5. 평균 제곱근 오차
# 오차를 평가하는 방법 중 가장 많이 사용되는 평균 제곱근 오차



## 6. 잘못 그은 선 바로잡기
# 평균 제곱 오차(Mean Squared Error, MSE)
# 평균 제곱근 오차(Root Mean Squared Error, RMSE)



## 7. 코딩으로 확인하는 평균 제곱근 오차

#기울기 a와 y 절편 b
ab=[3,76]                                     # 기울기 a = 3, y 절편 b = 76

# x,y의 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]   # 실제 데이터
x = [i[0] for i in data]                      # x값만 리스트로 변수 저장
y = [i[1] for i in data]                      # y값만 리스트로 변수 저장

# y=ax + b에 a,b 값 대입하여 결과를 출력하는 함수
def predict(x):
   return ab[0]*x + ab[1]

# RMSE 함수
def rmse(p, a):
   return np.sqrt(((p - a) ** 2).mean())

# RMSE 함수를 각 y값에 대입하여 최종 값을 구하는 함수
def rmse_val(predict_result,y):
   return rmse(np.array(predict_result), np.array(y))

# 예측값이 들어갈 빈 리스트
predict_result = []

# 모든 x값을 한 번씩 대입하여 predict_result 리스트완성.
for i in range(len(x)):
   predict_result.append(predict(x[i]))
   print("공부시간=%.f, 실제점수=%.f, 예측점수=%.f" % (x[i], y[i], predict(x[i])))

# 최종 RMSE 출력
print("rmse 최종값: " + str(rmse_val(predict_result,y)))
