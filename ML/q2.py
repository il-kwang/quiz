import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 
"""
1. data 폴더 내에 있는 dataset.csv파일을 불러오고, 
   학습용 데이터와 테스트용 데이터를 분리하여 
   반환하는 함수를 구현합니다.
"""
# 데이터 위치 : 

path = 'C:/Users/302/Documents/QuizFolder/quiz/ML/data/dataset.csv'
df = pd.read_csv(path)
print(df.head())

# 클래스가 0이랑 1로 구분 됨 
# 클래스 0 과 1의 구분 기준 찾기기



def load_data():
    
    df = pd.read_csv(path)
    
    X = df.drop('Class', axis = 1)
    y = df['Class']
    #데이터 분리 
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)
    print(X, y)
    return train_X, test_X, train_y, test_y
    
"""
2. SVM 모델을 불러오고,
   학습용 데이터에 맞추어 학습시킨 후, 
   테스트 데이터에 대한 예측 결과를 반환하는 함수를
   구현합니다.
"""



def SVM(train_X, test_X, train_y, test_y):
    
    svm = SVC(kernel='linear')
    
    # 학습용 데이터에 맞추어 모델을 학습시킵니다.
    svm.fit(train_X, train_y)
    
    pred_y = svm.predict(test_X)
    
    return pred_y
    
# 데이터를 불러오고, 모델 예측 결과를 확인하는 main 함수입니다.
def main():
    
    train_X, test_X, train_y, test_y = load_data()
    
    pred_y = SVM(train_X, test_X, train_y, test_y)
    
    # SVM 분류 결과값을 출력합니다.
    print("\nConfusion matrix : \n",confusion_matrix(test_y,pred_y))  
    print("\nReport : \n",classification_report(test_y,pred_y)) 

if __name__ == "__main__":
    main()
    
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Class'])
plt.xlabel('특성 1')
plt.ylabel('특성 2')
plt.title('데이터 시각화')
plt.show()

plt.scatter(df.iloc[:, 2], df.iloc[:, 3], c=df['Class'])
plt.xlabel('특성 3')
plt.ylabel('특성 4')
plt.title('특성 3과 4의 산점도')
plt.show()