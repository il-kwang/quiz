import numpy as np
import pandas as pd

DATA_PATH = "./250314-Exam2-pik/data/data.csv"

# 1. 데이터 불러오기
df = pd.read_csv(DATA_PATH)

# 2. 결측값 처리
df = df.dropna()

# 3. 지시사항 추가
df['Type'] = np.where(df['Age'] < 19, 'kid', 'adult')

#4. 출력 
print(df.head())

'''
# 6. 특성 엔지니어링

다른 칼럼의 정보를 이용하여 새로운 칼럼을 추가하는 방법을 실습합니다.

------

**지시사항과 기본적으로 주어진 코드의 주석을 참고**하여 `None` 부분을 채워 함수를 완성하세요.

지시사항에서 설명하는 함수 외의 코드를 **임의로 수정하는 경우**, 정상적으로 채점이 이루어지지 않을 수 있습니다.

모든 이름은 **대소문자를 구분**합니다.

------

**Hint**

```
numpy.where(조건, 값1, 값2)
```

- `조건`을 만족하는 데이터를 `값1`로, 그렇지 않은 데이터를 `값2`로 변환하는 함수입니다.

```
pd.DataFrame.loc[행 조건, 열 조건]
```

- `행 조건`과 `열 조건`을 만족하는 행렬을 반환하는 함수입니다.
- 예를 들어 `df.loc[df["Type"] == "kid", "Type"] = "아이"`는 Type이 kid인 사람들의 Type 값을 “아이”로 변경하는 코드입니다.

## **지시사항**

1. `add_type(df: pd.DataFrame)`: 새로운 특성 생성

   주어진 `df`에는 나이(`Age`)와 성별(`Sex`)가 포함된 데이터입니다.

   아래의 조건을 만족하는 새로운 칼럼 `Type`를 추가하고 반환하는 함수를 완성하세요.

   - 나이가 19세 이상이면 “adult”로 설정합니다.
   - 나이가 19세 미만이면서 여성(female)이라면 “girl”로 설정합니다.
   - 나이가 19세 미만이면서 남성(male)이라면 “boy”로 설정합니다.'''

def get_data() -> pd.DataFrame:
    "데이터를 불러오는 함수"

    df = pd.read_csv(DATA_PATH)
    return df



def add_type(df: pd.DataFrame) -> pd.DataFrame:
    "지시사항에 따라 df에 Type칼럼을 추가하고 반환합니다."

    df['Type'] = np.where(df['Age'] < 19, 'kid', 'adult')
    

    return df

def add_type(df: pd.DataFrame) -> pd.DataFrame:
    df['Type'] = np.where(df['Age'] < 19, 
                          np.where(df['Sex'] == 'female', 'girl', 'boy'), 
                          'adult')
    return df
   # 나이가 19세 미만이면서 여성(female)이라면 “girl”로 설정합니다.
   # 나이가 19세 미만이면서 남성(male)이라면 “boy”로 설정합니다

def main():
    # 데이터 불러오기
    df = get_data()
    print("추가 전\n", df.head())

    # 1. 새로운 특성 생성
    df_new = add_type(df.copy())
    print("추가 후\n", df_new.head())


if __name__ == "__main__":
    main()
