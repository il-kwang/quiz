import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# 데이터 주소
DATA_PATH = "./250314-Exam2-pik/data/taxi_fare_data.csv"

#데이터를 DataFram의 형태로 불러옵니다.
def load_csv(path):
    data_frame = pd.read_csv(path)
    return data_frame

# 결측치 처리 함수입니다.
def del_missing(df):
    
    # df에서 Unnamed: 0 feature 데이터를 제거하고 del_un_df에 저장합니다.
    del_un_df = df.drop(['Unnamed: 0'], axis='columns')
    
    # del_un_df에서 id feature 데이터를 제거하고 del_un_id_df에 저장합니다.
    del_un_id_df = del_un_df.drop(['id'], axis='columns')
    
    # del_un_id_df의 누락된 데이터가 있는 행을 제거하고 removed_df에 저장합니다.
    removed_df = del_un_id_df.dropna()
    
    return removed_df

# 리스트를 입력으로 받아서 해당 리스트 내에 음수값이 있으면 그 위치(인덱스)들을 리스트로 출력하는 함수를 만듭니다.
def get_negative_index(list_data):
    neg_idx = []
    
    for i, value in enumerate(list_data):
        if value < 0:
            neg_idx.append(list_data.index[i])
            
    return neg_idx

# DataFrame 내에 제거해야 하는 이상치의 인덱스를 반환하는 함수를 만듭니다.
def outlier_index():
    # get_negative_index() 함수를 통해서, fare_amount와 passenger_count 내의 음수값들의 인덱스를 반환합니다.
    idx_fare_amount = get_negative_index(fare_amount)
    idx_passenger_count = get_negative_index(passenger_count)
    
    idx_zero_distance = []    
    idx = [i for i in range(len(passenger_count))]
    zipped = zip(idx, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)
    
    for i, x, y, _x, _y in zipped:
        # 타는 곳(pickup_longitude,pickup_latitude)과 내리는 곳(drop_longitude, drop_latitude)이 같은 데이터의 인덱스를 idx_zero_distance에 저장합니다.
        if (x == _x) and (y == _y):
            idx_zero_distance.append(i)
            
    total_index4remove = list(set(idx_fare_amount+idx_passenger_count+idx_zero_distance))
    
    return total_index4remove

# 인덱스를 기반으로 DataFrame 내의 데이터를 제거하고, 제거된 DataFrame을 반환하는 함수를 만듭니다.
def remove_outlier(dataframe, list_idx):
    return dataframe.drop(list_idx)

# load_csv 함수를 사용하여 데이터를 불러와 df에 저장합니다.
df = load_csv(DATA_PATH)


#,id,pickup_datetime,pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,passenger_count,fare_amount
# id,픽업시간,픽업위도,픽업경도,도착위도,도착경도,인원수,요금

# 1-1. del_missing 함수로 df의 결측치을 처리하여 df에 덮어씌웁니다.
df = del_missing(df)
#결측치 처리 후 출력
print(df.head(10))
# 결측치 확인 

# 불러온 DataFrame의 각 인덱스의 값들을 변수로 저장합니다.
fare_amount = df['fare_amount']
passenger_count = df['passenger_count']
pickup_longitude = df['pickup_longitude']
pickup_latitude = df['pickup_latitude']
dropoff_longitude = df['dropoff_longitude']
dropoff_latitude = df['dropoff_latitude']

# 1-2. remove_outlier()을 사용하여 이상치를 제거합니다.
# remove_outlier()가 어떤 인자들을 받는지 확인하세요.
remove_index = outlier_index()
df = remove_outlier(df, remove_index)

# 2. df.corr()을 사용하여 상관 계수 값 계산
df = df[['fare_amount', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]
corr_df = df.corr()

# seaborn을 사용하여 heatmap 출력
plt.figure(figsize=(15,10))
sns.heatmap(corr_df, annot=True, cmap='PuBu')
plt.savefig("plot.png")


# regplot함수의 출력
plt.figure(figsize = (15, 10))
sns.regplot(x='fare_amount', y='passenger_count', data=df)
plt.savefig("plot2.png")





'''해결 안됨 아직직

#필요한 데이터의 NaN 제거
pickup_latitude = pickup_latitude.dropna()
pickup_longitude = pickup_longitude.dropna()
dropoff_latitude = dropoff_latitude.dropna()
dropoff_longitude = dropoff_longitude.dropna()

#픽업 위치와 도착 위치 사이의 거리 구하기
def distance(x1, y1, x2, y2): #함수 설명: x1, y1는 픽업 위도와 경도 y1,y2는 도착 위도와 경도
    return np.sqrt((x1-x2)**2 + (y1-y2)**2) # 픽업 위치와 도착 위치 사이의 거리 구하기 = 픽업위도 - 도착위도 제곱 + 픽업 경도 - 도착경도 제곱

# 거리 출력 테스트
print("distance 6번째 고객",distance(pickup_latitude[6], pickup_longitude[6], dropoff_latitude[6], dropoff_longitude[6]))

print("distance 1번째 고객",distance(pickup_latitude[0], pickup_longitude[0], dropoff_latitude[0], dropoff_longitude[0]))






# distance() 함수를 이용하여 거리 구하기
distance_list = [] 
for i in range(len(pickup_latitude)):
    distance_list.append(distance(pickup_latitude[i], pickup_longitude[i], dropoff_latitude[i], dropoff_longitude[i]))
    print(distance_list.append(distance(pickup_latitude[i], pickup_longitude[i], dropoff_latitude[i], dropoff_longitude[i])))

# distance() 함수를 이용하여 거리 구하기
distance_list = [] 
for i in range(len(pickup_latitude)):
    distance_list.append(distance(pickup_latitude[i], pickup_longitude[i], dropoff_latitude[i], dropoff_longitude[i]))

# distance_list를 df에 추가
df['distance'] = distance_list

# 구한 결과를 그래프로 출력 하고 그래프로 저장 x축 이동거리에 따른 y축 요금의 상관 관계계
plt.figure(figsize=(15,10))
plt.title("이동거리와 요금")
sns.regplot(x='distance', y='fare_amount', data=df)
plt.savefig("plot3.png")'''