### chapter 12. 다중 분류 문제 해결하기

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



## 1. 다중 분류 문제
# 샘플수: 150
# 속성: 4
    # 정보1 꽃받침 길이 (sepal length, 단위: cm)
    # 정보2 꽃받침 넓이 (sepal width, 단위: cm)
    # 정보3 꽃잎 길이 (petal length, 단위: cm)
    # 정보4 꽃잎 길이 (petal width, 단위: cm)
# 클래스: Iris-setosa, Iris-versicolor, Iris-virginica



## 2. 상관도 그래프
df = pd.read_csv('Dataset/iris.csv',
                 names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
df.head()
df.info()
df.describe()

sns.pairplot(df, hue = 'species')
plt.show()
# 판단: 꽃잎과 꽃받침의 크기와 넓이가 품종별로 차이가 있는것으로 보인다



## 3. 원-핫 인코딩
# Y의 값을 0과 1로만 이루어진 형태로 만드는 것
df = pd.read_csv('Dataset/iris.csv',
                 names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

dataset = df.values
X = dataset[:, :4].astype(float)
Y_obj = dataset[:, 4]

# 클래스의 이름을 숫자 형태로 바꿔주어야 함
from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 활성화 함수를 적용하려면 Y값이 숫자 0과 1로 이루어져 있어야 함
from keras.utils import np_utils
Y_encoded = np_utils.to_categorical(Y)



## 4. 소프트맥스
# 최종 출력 값이 3개 중 하나여야 하므로 출력층에 해당하는 Dense의 노드 수를 3으로 설정
# 활성화 함수로 앞서 나오지 않았던 소프트맥스(softmax)를 이용
model = Sequential()
model.add(Dense(16, input_dim = 4, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))



## 5. 아이리스 품좀 예측실행
# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 입력
df = pd.read_csv('Dataset/iris.csv',
                 names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# 그래프 확인
sns.pairplot(df, hue = 'species')
plt.show()

# 데이터 분류
dataset = df.values
X = dataset[:, :4]
Y_obj = dataset[:, 4]

# 문자열을 숫자로 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = np_utils.to_categorical(Y)

# 모델의 설정
model = Sequential()
model.add(Dense(16, input_dim = 4, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

# 모델 컴파일
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 실행
model.fit(X, Y_encoded, epochs = 50, batch_size = 1)

# 결과 출력
print('\n Accuracy: %.4f' % (model.evaluate(X, Y_encoded)[1]))