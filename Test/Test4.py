### 시험 3

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential   # 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.layers import Dense

if type(tf.contrib) != type(tf):      # warning 출력 안하기
    tf.contrib._warning = None



## 문제 1 ##################################################
## 1)
one = np.arange(0, 5, 0.5)
one

## 2)
two = np.arange(1, 11).reshape(2, 5)
two

## 3)
three = np.eye(N = 3, M = 3, dtype = np.int8)
three

## 4)
four = np.arange(16).reshape(4, 4)
np.ones_like(four)

## 5)
five = np.arange(1, 24, 2).reshape(3, 4)
five



## 문제 2 ##################################################
data = {
    "2015": [9904312, 3448737, 2890451, 2466052],
    "2010": [9631482, 3393191, 2632035, 2431774],
    "2005": [9762546, 3512547, 2517680, 2456016],
    "2000": [9853972, 3655437, 2466338, 2473990],
    "지역": ["수도권", "경상권", "수도권", "경상권"]}
columns = ["지역", "2015", "2010", "2005", "2000"]
index = ["서울", "부산", "인천", "대구"]
df = pd.DataFrame(data, index=index, columns=columns)
df['2010~2015 증가율'] = (df['2015'] - df['2010']) / df['2010']
df



## 문제 3 ##################################################
import seaborn as sns
titanic = sns.load_dataset("titanic")    # 타이타닉호 데이터

## 1) 성별인원수, 선실별인원수, 사망/생존인원수 구하기
titanic.groupby('sex')['sex'].count()
titanic.groupby('class')['class'].count()
titanic.groupby('alive')['alive'].count()

## 2)######################################################################
titanic['age_group'] = pd.qcut(titanic['age'], 5, labels = ['미성년자', '청년', '중년', '장년', '노년'])
titanic[['age', 'age_group']]



## 문제 4 ##################################################
import seaborn as sns
tips = sns.load_dataset("tips")    # 팁 데이터
tips.head()

## 1) 팁의 비율
tips['팁비율'] = (tips['tip'] / tips['total_bill'] * 100).round(2)

## 2)
data = pd.pivot_table(tips,
                      index = 'day',
                      aggfunc = 'mean')
data = data.sort_values(by = '팁비율', ascending = False)
data['팁비율']



## 문제 5 ##################################################
import sqlite3
conn = sqlite3.connect('Test/Eagles1.db')
'''
conn = sqlite3.connect(':memory:')    # 메모리 DB 접속(일회성)
'''

## 1) 필드로 (백넘버, 이름,포지션)을 갖는 테이블 만들기
cur = conn.cursor()
cur.execute('CREATE TABLE IF NOT EXISTS Eagles1 \
    (backNo INT NOT NULL, \
     name TEXT, \
     position TEXT, \
     PRIMARY KEY(backNo));')

## 2) (8,정근우, 내야수)포함하여 임의로 5명의 선수 삽입
cur = conn.cursor()
cur.execute("INSERT INTO Eagles1 VALUES(8, '정근우', '내야수');")
cur.execute("INSERT INTO Eagles1 VALUES(10, '이태양', '투수');")
cur.execute("INSERT INTO Eagles1 VALUES(1, '김태균', '내야수');")
cur.execute("INSERT INTO Eagles1 VALUES(11, '이성열', '외야수');")
cur.execute("INSERT INTO Eagles1 VALUES(12, '정우람', '투수');")
conn.commit()

## 3)
cur = conn.cursor()
cur.execute('SELECT * FROM Eagles1')
for row in cur:
    print(row)

## 4) 정근우 포지션 외야수로 변경
## UPDATE table SET field1 = value1, ... WHERE 조건;
cur = conn.cursor()
cur.execute("UPDATE Eagles1 SET position = '외야수' WHERE backNo = 8;")
conn.commit()

## 5) 5명 중 백넘거 가장 큰 선수 삭제
cur = conn.cursor()

# 조회
cur.execute('SELECT * FROM Eagles1 ORDER BY backNo DESC')
rows = cur.fetchall();
for row in rows:
    print(row)

# 12번 삭제
cur.execute("DELETE FROM Eagles1 WHERE backNo = 12;")
conn.commit()
conn.close()



## 문제 6 ##################################################
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

X = np.linspace(-3, 3, 1001)

sig_Y = sigmoid(X)
derivative_Y = derivative_sigmoid(X)

plt.plot(X, sig_Y, color='b')
plt.plot(X, derivative_Y, color='r')
plt.title('Sigmoid')
plt.show()


## 문제 7 ##################################################

## 1) 경사 하강법

## 2) 고급 경사 하강법

## 3) 퍼셉트론

## 4) relu

## 5) 은닉층



## 문제 8 ##################################################
import seaborn as sns
iris = sns.load_dataset("iris")    # 팁 데이터
iris.head()
iris.info()

# X에는 속성, Y_obj에는 클래스 지정
dataset = iris.values
X = dataset[:, :4]
Y_obj = dataset[:, 4]

# 문자열 숫자로 바꿔줌
from sklearn.preprocessing import LabelEncoder
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 숫자를 0과 1로 바꿔줌
from keras.utils import np_utils
Y_encoded = np_utils.to_categorical(Y)

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 모델 설정
# 은닉층1, 입려층 : 노드 16, 입력4, relu
# 은닉층2        : 노드 8, relu
# 출력층         : 3, softmax
model = Sequential()
model.add(Dense(16, input_dim = 4, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

# 모델 컴파일
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 실행
model.fit(X, Y_encoded, epochs = 100, batch_size = 10)

# 결과 출력
print('\n Accuracy: %.4f' % (model.evaluate(X, Y_encoded)[1]))



## 문제 9 ##################################################
df = pd.read_csv('Test/pima-indians-diabetes.csv', header = None)
df.info()
df.head()

dataset = df.values
X = dataset[:, :8]
Y = dataset[:, 8]

# 학습셋과 테스트셋을 나눔
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = seed)

# 모델의 설정
model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일
model.compile(loss = 'binary_crossentropy',
              optimizer= 'adam',
              metrics= ['accuracy'])

# 모델 실행
model.fit(X_train, Y_train, epochs = 200, batch_size = 10)

# 모델을 컴퓨터에 저장
model.save('Test/model1.h5')

# 테스트를 위해 메모리 내의 모델을 삭제
del model

# 모델을 새로 불러옴
from keras.models import Sequential, load_model
model = load_model('Test/model1.h5')

# 결과 출력
print('\n Accuracy: %.4f' % (model.evaluate(X_test, Y_test)[1]))