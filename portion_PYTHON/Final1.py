from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename
import subprocess
import os
import shutil  # 파일 복사를 위한 모듈
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'C:/Y5/yolov5-master/processing'  # 업로드된 파일이 저장될 디렉토리
PROCESSED_FOLDER = 'C:/Y5/yolov5-master/processed'  # 가공 후 앱으로 보낼 가공 영상 저장
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 파일 저장
    desired_filename = "Yolotest.mp4"
    filename = secure_filename(desired_filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)

    # detect.py 실행
    command = ["python", "detect3.py"]
    subprocess.run(command)

    # detect.py에 의해 생성된 output.mp4 파일 경로
    generated_video_path = 'C:/Y5/yolov5-master/result/output.mp4'

    # 결과 파일을 processed 폴더로 복사
    if not os.path.exists(app.config['PROCESSED_FOLDER']):
        os.makedirs(app.config['PROCESSED_FOLDER'])

    processed_video_path = os.path.join(app.config['PROCESSED_FOLDER'], "output.mp4")
    shutil.copyfile(generated_video_path, processed_video_path)

    if not os.path.exists(processed_video_path):
        return jsonify({'error': 'Processed video not found'}), 500

    print("'processed' 폴더에 가공된 영상 저장 완료")

    time.sleep(1)

    def generate():
        try:
            with open(processed_video_path, 'rb') as f:
                while True:
                    chunk = f.read(1024)
                    if not chunk:
                        print("반환 완료")
                        break
                    yield chunk
        except Exception as e:
            print(f"파일 읽기 에러: {e}")
            return jsonify({'error': 'File reading error'}), 500

    return Response(generate(), content_type='video/mp4'), 200

if __name__ == '__main__':
    app.run('0.0.0.0', port=8080, debug=True)
