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
path = './250314-Exam2-pik/data/train.csv' #./250314-Exam2-pik/data/taxi_fare_data.csv

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
#plt.show()

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
#plt.show()

#사이킷런 구현하기
x = df[['distance']]
y = df['fare_amount']

lr = LinearRegression(fit_intercept=True)
lr.fit(x,y)
lr.fit(df[['distance']], df['fare_amount'])
print('절편 :',lr.intercept_)
print('회귀 계수 :',lr.coef_)
print(lr.score(df[['distance']],df['fare_amount']))

#이상치 판단
#이동 거리가 0인 데이터 제거하기
df = df[df['distance'] != 0]

#이동 거리와 지불 가격의 상관관계 구하기
plt.scatter(df['distance'], df['fare_amount'])
plt.xlabel('이동 거리')
plt.ylabel('지불 가격') 
plt.savefig('이동거리 이상치 추가 제거.png')
#plt.show()

#사이킷런 구현하기
x = df[['distance']]
y = df['fare_amount']

lr = LinearRegression(fit_intercept=True)
lr.fit(x,y)
lr.fit(df[['distance']], df['fare_amount'])
print('절편 :',lr.intercept_)
print('회귀 계수 :',lr.coef_)
print(lr.score(df[['distance']],df['fare_amount']))

def get_negative_index(df):  #이상치 판단 함수 음수값을 찾는 함수
    return df[df < 0].index 


# 이상치 제거 함수입니다.
# 이 함수는 데이터프레임에서 이상치 데이터를 찾아 제거합니다.
def outlier_index(df):
    # fare_amount 열에서 음수 값을 갖는 인덱스를 찾습니다.
    # 음수 값은 이상치 데이터로 간주합니다.
    idx_fare_amount = get_negative_index(df['fare_amount']) 
    
    # passenger_count 열에서 음수 값을 갖는 인덱스를 찾습니다.
    # 음수 값은 이상치 데이터로 간주합니다.
    idx_passenger_count = get_negative_index(df['passenger_count'])
    
    # pickup_longitude와 dropoff_longitude가 같고, pickup_latitude와 dropoff_latitude가 같은 데이터의 인덱스를 찾습니다.
    # 이 경우는 거리가 0이라는 것을 의미하므로, 이상치 데이터로 간주합니다.
    idx_zero_distance = df[
        (df['pickup_longitude'] == df['dropoff_longitude']) & 
        (df['pickup_latitude'] == df['dropoff_latitude'])
    ].index
    
    # 이상치 데이터의 인덱스를 결합하여 중복을 제거하고, 반환합니다.
    # 반환할 때 ValueError가 발생하는 것을 방지하기 위해 인덱스를 리스트로 변환합니다.
    return list(set(idx_fare_amount.tolist() + idx_passenger_count.tolist() + idx_zero_distance.tolist()))

print(outlier_index(df))

#위에 이상치 제거 중 애러 발생생

def remove_outlier(df, list_idx):#이상치 제거 함수
    return df.drop(index=list_idx) # 이상치 제거

#이상치 제거 후 상관관계 구하기
df = remove_outlier(df, outlier_index(df)) 
plt.scatter(df['distance'], df['fare_amount'])
plt.xlabel('이동 거리')
plt.ylabel('지불 가격')
plt.savefig('이상치 제거 후 상관관계 구하기.png') #
#plt.show() 

#사이킷런 구현하기
x = df[['distance']]
y = df['fare_amount']

lr = LinearRegression(fit_intercept=True)
lr.fit(x,y)
lr.fit(df[['distance']], df['fare_amount'])
print('절편 :',lr.intercept_)
print('회귀 계수 :',lr.coef_)
print(lr.score(df[['distance']],df['fare_amount']))

'''
#DBSSCAN 알고리즘을 이용한 클러스터링
from sklearn.cluster import DBSCAN
from sklearn import metrics

# 날짜형 데이터를 제거합니다.
df = df.drop(['pickup_datetime', 'dropoff_datetime'], axis=1)

# DBSCAN 알고리즘을 적용합니다.
dbscan = DBSCAN(eps=0.5, min_samples=10)

# 데이터를 클러스터링합니다.
clusters = dbscan.fit_predict(df)

# 이상치 데이터를 식별합니다.
dbscan_outliers = df[clusters == -1]

# 이상치 데이터를 제거합니다.
df_clean = df[clusters != -1]

# 성능 평가하기
print('DBSCAN 성능 평가')
print('DBSCAN 성능 평가 :', metrics.silhouette_score(df_clean, clusters))
print('DBSCAN 성능 평가 :', metrics.calinski_harabaz_score(df_clean, clusters))
print('DBSCAN 성능 평가 :', metrics.davies_bouldin_score(df_clean, clusters))

#이상치 제거 후 상관관계 구하기
plt.scatter(df_clean['distance'], df_clean['fare_amount'])
plt.xlabel('이동 거리')
plt.ylabel('지불 가격')
plt.savefig('DBSCAN 이상치 제거 후 상관관계 구하기.png')
'''



