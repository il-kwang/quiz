from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# 모델 로드
try:
    # 현재 파일의 디렉토리 경로를 기준으로 모델 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'lgbm_model.pkl')
    model = joblib.load(model_path)
    print("모델이 성공적으로 로드되었습니다.")
except FileNotFoundError:
    print("모델 파일을 찾을 수 없습니다. 경로를 확인하세요.")
    exit(1)
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 클라이언트로부터 JSON 데이터 받기
        data = request.get_json()
        df = pd.DataFrame(data)  # 데이터를 Pandas DataFrame으로 변환
        
        # 모델로 예측 수행
        prediction = model.predict(df)
        print("예측 결과:", prediction)  # 서버 로그에 예측 결과 출력
        
        # 예측 결과를 JSON 형식으로 반환
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        # 오류 발생 시 오류 메시지 반환
        return jsonify({'error': str(e)}), 400

@app.route('/hi', methods=['GET'])
def hi():
    return 'hello'

if __name__ == '__main__':
    try:
        app.run('127.0.0.1', port=5000, debug=True)
    except Exception as e:
        print(f"Flask 서버 실행 중 오류 발생: {e}")