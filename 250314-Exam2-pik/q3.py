import pandas as pd
import numpy as np

# 데이터 주소
DATA_PATH = './250314-Exam2-pik/data/taxi_fare_data.csv'

# pandas를 이용하여 데이터를 DataFrame의 형태로 불러오는 load_csv 함수를 설정합니다.
def load_csv(path):
    data_frame = pd.read_csv(path)
    return data_frame

# load_csv 함수를 사용하여 데이터를 불러와 df에 저장합니다.
df = load_csv(DATA_PATH)

print("누락된 데이터(Missing Data)를 제거하기 전의 데이터 정보")
df.info()
print(df)
print('--'*50)

# df에서 "Unnamed: 0" 컬럼을 제거하고 del_un_df에 저장합니다.
del_un_df = df.drop('Unnamed: 0',axis=1)
print(del_un_df)

# del_un_df에서 "id" 컬럼을 제거하고 del_un_id_df에 저장합니다.
del_un_id_df = del_un_df.drop('id',axis=1)
print(del_un_id_df)

# del_un_id_df의 결측치가 포함된 행을 제거하고 removed_df에 저장합니다.
removed_df = del_un_id_df.dropna(axis=0)
print("\n결측치(Missing Data)를 제거한 후의 데이터 정보")
print(removed_df)
removed_df.info()
