import requests
import pandas as pd

# 1. 서버 URL
url = 'http://127.0.0.1:5000/predict'

# 2. CSV 파일 읽기
#C:/Users/302/Documents/미래융합교육원/MLdata/creditcard.csv
#C:/Users/302/Documents/미래융합교육원/quiz/ML/creditcard/X_train_over_no_target.csv
csv_path = "C:/Users/302/Documents/미래융합교육원/quiz/ML/creditcard/X_train_over_100_target.csv"

csv_path_home = "C:/Users/user/OneDrive/문서/미래융합교육원/quiz/ML/creditcard/X_train_over_100_target.csv"

data = pd.read_csv(csv_path_home)

# 3. 입력 데이터 준비
input_data = data.to_dict(orient='records')

# 4. 서버에 요청 보내기
try:
    res = requests.post(url, json=input_data)

    # 서버 응답 처리
    if res.status_code == 200:
        predictions = res.json()['prediction']
        print("예측 결과:", predictions)

        # 0과 1의 개수 출력
        count_0 = predictions.count(0)
        count_1 = predictions.count(1)
        print(f"0의 개수: {count_0}, 1의 개수: {count_1}")
    else:
        print(f"오류 발생: {res.status_code}, {res.text}")
except Exception as e:
    print(f"요청 중 오류 발생: {e}")