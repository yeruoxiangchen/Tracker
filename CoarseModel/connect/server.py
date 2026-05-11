import os
import shutil
import time
import json
import subprocess
from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np

app = Flask(__name__)

# 统一目录与信号文件配置
BASE_DIR = "/home/zjr/ReconViaGen/ar_tracker"
DATA_DIR = os.path.join(BASE_DIR, "data")
PREVIEW_DIR = os.path.join(BASE_DIR, "previews")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FLAG_DIR = os.path.join(BASE_DIR, "flags")

FLAG_START_PREPROCESS = os.path.join(FLAG_DIR, "start_preprocess.flag")
FLAG_PREPROCESS_DONE = os.path.join(FLAG_DIR, "preprocess_done.json")
FLAG_START_GENERATE = os.path.join(FLAG_DIR, "start_generate.json")
FLAG_GENERATE_DONE = os.path.join(FLAG_DIR, "generate_done.flag")

frame_counter = 0

def clean_environment():
    # 注意：从清理列表中移除了 OUTPUT_DIR，防止删除以前生成的 3D 模型
    for d in [DATA_DIR, PREVIEW_DIR, FLAG_DIR]:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        
    # OUTPUT_DIR 只要确保存在即可，绝对不要清空它
    os.makedirs(OUTPUT_DIR, exist_ok=True)
@app.route('/start_record', methods=['POST'])
def start_record():
    global frame_counter
    print("\n>>> [调度] 收到开始录制请求，正在清理环境...")
    clean_environment()
    frame_counter = 0
    return jsonify({"status": "ready"}), 200

@app.route('/upload', methods=['POST'])
def upload():
    global frame_counter
    try:
        pos_x, pos_y, pos_z = request.form.get('pos_x'), request.form.get('pos_y'), request.form.get('pos_z')
        rot_x, rot_y, rot_z = request.form.get('rot_x'), request.form.get('rot_y'), request.form.get('rot_z')

        img_bytes = request.files['image'].read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        frame_name = f"frame_{frame_counter:04d}.jpg"
        cv2.imwrite(os.path.join(DATA_DIR, frame_name), img)
        
        with open(os.path.join(DATA_DIR, "poses.txt"), "a") as f:
            f.write(f"{frame_name},{pos_x},{pos_y},{pos_z},{rot_x},{rot_y},{rot_z}\n")
        
        frame_counter += 1
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/preprocess', methods=['POST'])
def preprocess():
    print("\n>>> [调度] 停止录制，通知后台开始抠图...")
    open(FLAG_START_PREPROCESS, 'w').close()
    
    for _ in range(120):
        if os.path.exists(FLAG_PREPROCESS_DONE):
            with open(FLAG_PREPROCESS_DONE, 'r') as f:
                data = json.load(f)
            os.remove(FLAG_PREPROCESS_DONE)
            print(f">>> [调度] 抠图完毕，通知手机拉取 {data['total']} 张预览图")
            return jsonify({"status": "preprocessed", "total_images": data['total']}), 200
        time.sleep(1)
        
    return jsonify({"status": "error", "message": "后台抠图超时"}), 500

@app.route('/get_preview/<int:img_id>', methods=['GET'])
def get_preview(img_id):
    return send_from_directory(PREVIEW_DIR, f"{img_id}.png")

# ================= 新增：取消并重置接口 =================
@app.route('/cancel_review', methods=['POST'])
def cancel_review():
    global frame_counter
    print("\n>>> [调度] 收到手机端【重置/退出】指令！强制中断并重置管线...")
    
    # 给 run_local 发送一个空列表，让它立刻中断当前的生成并退回开头
    with open(FLAG_START_GENERATE, 'w') as f:
        json.dump({"selected": []}, f)
        
    # ⚠️ 极其关键：暂停 0.5 秒，给后台的 run_local.py 一点点时间去读取上面的中断指令
    time.sleep(0.5) 
    
    clean_environment()
    frame_counter = 0
    return jsonify({"status": "cancelled"}), 200
# ==============================================================

@app.route('/generate', methods=['POST'])
def generate():
    selected_indices = request.json.get('selected', [])
    print(f"\n>>> [调度] 收到手机发来的保留序号 {selected_indices}，通知后台开始最终生成...")
    
    with open(FLAG_START_GENERATE, 'w') as f:
        json.dump({"selected": selected_indices}, f)
        
    for _ in range(600):
        if os.path.exists(FLAG_GENERATE_DONE):
            os.remove(FLAG_GENERATE_DONE)
            print(">>> [调度] 后台生成成功！结果已返回给手机。")
            return jsonify({"status": "success", "message": "Mesh Generated!"}), 200
        time.sleep(1)

    return jsonify({"status": "error", "message": "后台生成超时"}), 500

if __name__ == '__main__':
    clean_environment()
    print(f">>> [调度] 正在由 Server 自动唤醒后台 3D 重建进程...")
    script_path = "/home/zjr/Tracker/ReconViaGen/run_local.py"
    try:
        subprocess.Popen(
            [
                "conda", "run", "-n", "reconviagen", 
                "--no-capture-output", "python", "-u", script_path
            ], 
            cwd="/home/zjr/ReconViaGen"
        )
    except Exception as e:
        print(f"❌ 自动拉起 run_local.py 失败: {e}")
    
    print(f"服务器启动，作为调度器监听 5000 端口...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)