import pandas as pd

a = pd.Series([20, 15, 30, 25, 35], name='age')
b = pd.Series([68.5, 60.3, 53.4, 74.1, 80.7], name='weight')
c = pd.Series([180, 165, 155, 178, 185], name ='height')

## 1. human DataFrame 만들기
human = pd.DataFrame([a,b,c])
print(human,'\n')

def main():
    ### 2. loc(), iloc() 함수를 이용하여 `age`를 추출
    print(human.loc['age'],'\n')
    print(human.iloc[0],'\n')
    
    ### 3. loc(), iloc() 함수를 이용하여 `weight`와 `height`만 추출
    print(human.loc[['weight' ,'height']],'\n')
    print(human.iloc[1:3],'\n')
     
    sex = pd.Series(['F','M','F','M','F'], name = 'sex')
    ### 4.새로운 데이터 `sex`를 `human`에 추가
    human.loc['sex'] = sex
    print(human,'\n')
    
    ### 5. `human`에서 `height`를 삭제
    tmp = human.drop(['height'])
    print(tmp,'\n')


if __name__ == "__main__":
    main()
