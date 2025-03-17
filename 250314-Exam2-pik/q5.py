import numpy as np
import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv("./250314-Exam2-pik/data/taxi_fare_data.csv", quoting=3)

# UTC 제거 (혹시 남아 있는 공백도 정리)
df['pickup_datetime'] = df['pickup_datetime'].str.replace(" UTC", "", regex=False).str.strip()

# 변환 시 초단위 제거하고 적용 (초 제외)
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format="%Y-%m-%d %H:%M", errors='coerce')

# 변환 실패한 값 확인
print(df[df['pickup_datetime'].isna()])  # NaT 남아있는 데이터 확인

# 정상 변환 확인
print(df.info())
print(df[['pickup_datetime']].head(10))  # 변환된 데이터 샘플 확인

# 연, 월, 일, 시간 컬럼 추가
df['year'] = df['pickup_datetime'].dt.year
df['month'] = df['pickup_datetime'].dt.month
df['day'] = df['pickup_datetime'].dt.day
df['hour'] = df['pickup_datetime'].dt.hour

# 변환된 데이터 출력
print(df[['year', 'month', 'day', 'hour']].head(10))