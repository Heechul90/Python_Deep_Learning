### chapter 11. 데이터 다루기

# 함수 준비하기

from keras.models import Sequential  # 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.layers import Dense

# 모듈 준비하기

import numpy as np                # 필요한 라이브러리를 불러옵니다.
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

if type(tf.contrib) != type(tf):  # warning 출력 안하기
    tf.contrib._warning = None



## 1, 딥러닝과 데이터
# 머신러닝 프로젝트의 성공과 실패는 얼마나 좋은 데이터르르 가지고 시작하느냐에 영향을 많이 받음



## 2. 피마 인디언 데이터 분석하기
# 샘플수: 768
# 속성: 8
    # 정보1 (pregnant)  과거 임신 횟수
    # 정보2 (plasma)    포도당 부하 검사 2시간 후 공복 혈당 농도(mm Hg)
    # 정보3 (pressure)  확장기 혈압(mm Hg)
    # 정보4 (thickness) 삼두근 피부 주름 두께(mm)
    # 정보5 (insulin)   혈청 인슐린(2-hour, mu U/ml)
    # 정보6 (BMI)       체질량 지수(BMI, weight in kg/(height in m)**2)
    # 정보7 (pedigree)  당뇨병 가족력
    # 정보8 (age)       나이
# 클래스: 당뇨(1), 당뇨 아님(0)



## 3. pandas를 활용한 데이터 조사
df = pd.read_csv('Dataset/pima-indians-diabetes.csv',
                 names = ['pregnant', 'plasma', 'pressure', 'thickness', 'insulin', 'BMI', 'pedigree', 'age', 'class'])
df.head()
df.info()
df.describe()
df[['pregnant', 'class']]



## 4. 데이터 가공하기
# 임신횟수와 당뇨병 발병 확률
print(df[['pregnant', 'class']].groupby(['pregnant'],
      as_index = False).mean().sort_values(by = 'pregnant', ascending = True))



## 5. matplotlib를 이용해 그래프로 표현하기

# 상관관계를 나타내 주는 heatmap 함수를 통해 그래프 그려보기
plt.figure(figsize = (12, 12))
sns.heatmap(df.corr(),
            linewidths = 0.1,
            vmax = 0.5,                  # 색상의 밝기 조절
            cmap = plt.cm.gist_heat,     # 미리 정해진 matplotlib 색상의 설정값
            linecolor = 'white',
            annot = True)
plt.show()
# 판단: plasma와 class의 값이 0.47로 가장 높아 plasma가 당뇨병을 만드는데
#      가장 큰 역할을 한 것으로 보입니다.

grid = sns.FacetGrid(df, col = 'class')
grid.map(plt.hist, 'plasma', bins = 10)
plt.show()
# 판단: 당뇨병 환자의 경우 plasma가 150 이상인 경우가 많은 것으로 보입니다.



## 6. 피마 인디언의 당뇨병 예측 실행
# seed 값 생성
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = np.loadtxt('Dataset/pima-indians-diabetes.csv', delimiter = ',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

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
model.fit(X, Y, epochs = 200, batch_size = 10)

# 결과 출력
print('\n Accuracy: %.4f' % (model.evaluate(X, Y)[1]))