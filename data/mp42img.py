import cv2
import os

video_path = "/home/zjr/Tracker/data/wogua/wogua.mp4"
output_dir = "/home/zjr/Tracker/data/wogua/images"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0
save_idx = 0
save_stride = 10   # 每隔 5 帧保存一次，可以改成你需要的值

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % save_stride == 0:
        # 保存成 4 位数编号的文件名：0000.jpg, 0001.jpg, ...
        filename = os.path.join(output_dir, f"{save_idx:04d}.jpg")
        cv2.imwrite(filename, frame)
        save_idx += 1

    frame_idx += 1

cap.release()
print(f"提取完成，共保存 {save_idx} 帧到 {output_dir}，原视频共 {frame_idx} 帧")
