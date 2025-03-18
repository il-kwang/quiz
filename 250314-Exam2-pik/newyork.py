#뉴욕 택시요금  분석
'''
관측치 50000개
택시 요금 예측
하버사인으로 거리 계산, 마일로 변환 계산
사이킷런 선형회귀모델, OLS
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#뉴욕택시 데이터
path = './data/train.csv' #./250314-Exam2-pik/data/taxi_fare_data.csv

#뉴욕택시 데이터 불러오기
df = pd.read_csv(path)
print(df.head(10))

#결측치 확인
print(df.isna().sum()) #확인 시 결측치 없는거 확인

# key,fare_amount,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count
# 날짜, 요금, 픽업 데이터시간(타임스태프), 픽업 경도, 픽업 위도, 도착 경도, 도착 위도, 픽원인원수

#거리 구하고 마일로 변환하기
df['distance'] = ((df['pickup_longitude'] - df['dropoff_longitude']) ** 2 + (df['pickup_latitude'] - df['dropoff_latitude']) ** 2) ** 0.5
df['distance'] = df['distance'] * 0.000621371
print(df.head(10))

#distance(거리)와 지불 가격의 상관관계 구하기 - scatter plot 
plt.rc('font', family='Malgun Gothic')
plt.scatter(df['distance'], df['fare_amount'])
plt.xlabel('이동 거리')
plt.ylabel('지불 가격')
plt.savefig('가격과 거리의 상관관계.png')
plt.show()

#사이킷런 구현하기
#모델링 기본 구조
x = df[['distance']]
y = df['fare_amount']

lr = LinearRegression(fit_intercept=True)
lr.fit(x,y)
lr.fit(df[['distance']], df['fare_amount'])
print('절편 :',lr.intercept_)
print('회귀 계수 :',lr.coef_)
print(lr.score(df[['distance']],df['fare_amount'])) #모형 성능 평가가
#0.00017898865362730998
#가격과 거리의 상관관계가 이상함.;;


#이동 시간과 요금의 관계

#이상치 제거하기
#탑승 위치와 도착 위치가 같은 데이터 제거하기
df = df[df['pickup_longitude'] != df['dropoff_longitude']]
df = df[df['pickup_latitude'] != df['dropoff_latitude']]

#이상치 제거 후 상관 관계 구하기
plt.scatter(df['distance'], df['fare_amount'])
plt.xlabel('이동 거리')
plt.ylabel('지불 가격')
plt.savefig('가격과 지불 이동 거리의 상관관계 이상치 제거거.png')
plt.show()

#사이킷런 구현하기
x = df[['distance']]
y = df['fare_amount']

lr = LinearRegression(fit_intercept=True)
lr.fit(x,y)
lr.fit(df[['distance']], df['fare_amount'])
print('절편 :',lr.intercept_)
print('회귀 계수 :',lr.coef_)
print(lr.score(df[['distance']],df['fare_amount']))
