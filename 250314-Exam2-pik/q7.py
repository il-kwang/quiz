import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# 데이터 주소
DATA_PATH = "./250314-Exam2-pik/data/taxi_fare_data.csv"

# 데이터 불러오기
def load_csv(path):
    return pd.read_csv(path)

# 결측치 처리 함수
def del_missing(df):
    df = df.drop(['Unnamed: 0', 'id'], axis=1, errors='ignore')  # 해당 컬럼이 없어도 오류 방지
    return df.dropna()

# 음수값이 있는 인덱스 찾기
def get_negative_index(series):
    return series[series < 0].index.tolist()

# 이상치 인덱스 찾기
def outlier_index(df):
    idx_fare_amount = get_negative_index(df['fare_amount'])
    idx_passenger_count = get_negative_index(df['passenger_count'])

    idx_zero_distance = df[
        (df['pickup_longitude'] == df['dropoff_longitude']) & 
        (df['pickup_latitude'] == df['dropoff_latitude'])
    ].index.tolist()

    return list(set(idx_fare_amount + idx_passenger_count + idx_zero_distance))

# 이상치 제거 함수 (존재하는 인덱스만 삭제)
def remove_outlier(df, list_idx):
    valid_idx = list(set(df.index).intersection(set(list_idx)))  # 실제 존재하는 인덱스만 필터링
    return df.drop(index=valid_idx) if valid_idx else df  # 삭제할 인덱스가 없으면 그대로 반환

# 데이터 로드 및 처리
df = load_csv(DATA_PATH)
df = del_missing(df)

# 이상치 제거
remove_index = outlier_index(df)
df = remove_outlier(df, remove_index)

# 상관계수 계산 및 시각화
df = df[['fare_amount', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]
corr_df = df.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(corr_df, annot=True, cmap='PuBu')
plt.savefig("plot.png")

plt.figure(figsize=(15, 10))
sns.regplot(x='fare_amount', y='passenger_count', data=df)
plt.savefig("plot2.png")

print(" 데이터 전처리 완료!")
print("데이터 크기:", df.shape)
