타이타닉 캐글

# ML/titanic/titanic.md
먼지 모르겠는데 gender_submission.csv 파일 확인 시 0은 사망 1은 생존

- 생각 해 볼 수 있는거 
- 나이가 너무 어리거나 or 너무 많으면 생존율이 높을까?
- 성별에 따라 생존률이 다를까?
- 티켓 등급에 따라 생존률이 다를까?
- 탑승 항구에 따라 생존률이 다를까?


Survived: 생존 여부 => 0 = No, 1 = Yes
pclass: 티켓 등급 => 1 = 1st, 2 = 2nd, 3 = 3rd
Sex: 성별
Age: 나이  -   결측치 177개인데 사용을 할 수 있나 모르겠음음
Sibsp: 함께 탑승한 형제자매, 배우자의 수
Parch: 함께 탑승한 부모, 자식의 수
Ticket: 티켓 번호
Fare: 운임비용
Cabin: 객실 번호   - 안쓸 예정 
Embarked: 탑승 항구 => C = Cherbourg, Q = Queenstown, S = Southampton - 결측치 2개인데 버려야되나..

확인 #나이 177개 결측치, Cabin 687개 결측치, Embarked 2개 결측치

객실 번호는 안쓸 예정.
