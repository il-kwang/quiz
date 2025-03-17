# Seaborn 라이브러리 import 하기 
import seaborn as sns 
import matplotlib.pyplot as plt 

# seaborn의 load_dataset을 사용하여 tips (팁 가격) 데이터 불러오기
df = sns.load_dataset('tips') 

# countplot함수의 출력물을 sns_plot_size으로 저장
# x축을 "size" 컬럼으로 하여 "size"에 대한 countplot을 그림
sns_plot_size = None

# jointplot함수의 출력물을 g로 저장
# x축은 "total_bill", y축은 "tip", 차트의 종류는 "resid"으로 하여 jointplot을 그림
g = None

# 출력을 위한 함수
fig = sns_plot_size.get_figure()
fig.savefig("plot_siz.png")

g.savefig("plot.png")
