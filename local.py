import os
import cv2
import torch
import tkinter as tk
from tkinter import filedialog

# 모델 로드

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/exp5/weights/best.pt')
model.to(device)  # 모델을 GPU로 이동

# 동영상 처리 함수 (실시간 처리)
def process_video_real_time(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open video file.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델로 분석
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        results_img = results.render()[0]
        results_img_bgr = cv2.cvtColor(results_img, cv2.COLOR_RGB2BGR)

        # 결과를 화면에 실시간으로 출력
        cv2.imshow("Processed Video", results_img_bgr)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI를 통한 파일 선택 함수
def select_video_file():
    root = tk.Tk()
    root.withdraw()  # 기본 윈도우 숨기기

    # 파일 다이얼로그로 비디오 파일 선택
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
    )
    
    if video_path:
        print(f"Selected video: {video_path}")
        # 실시간으로 동영상 분석 시작
        process_video_real_time(video_path)

if __name__ == '__main__':
    select_video_file()
