import cv2

# 图像路径
img_path = "/home/zjr/Tracker/resize/RealSenseRecorder/3/color/1.jpg"

# 用于存储标记的点
points = []

# 鼠标回调函数
def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(img, f"{len(points)}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        cv2.imshow("image", img)
        print(f"Point {len(points)}: ({x}, {y})")

# 读取图像
img = cv2.imread(img_path)
cv2.imshow("image", img)

# 设置鼠标回调
cv2.setMouseCallback("image", click_event)

print("请依次点击：A4纸短边的两个角点，然后点击模型上需要测量的点。")
print("按 'q' 退出标记。")

# 循环显示图像，等待键盘退出
while True:
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

print("标记的所有点：", points)
