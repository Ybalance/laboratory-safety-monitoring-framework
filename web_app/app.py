import os
import cv2
import numpy as np
from flask import Flask, render_template, request, Response, jsonify, send_file
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import time
import threading
from datetime import datetime
import json
import sqlite3
import requests
from flask import g

app = Flask(__name__)

# 配置
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 最大上传

# VLM 配置
VLM_API_KEY = "sk-nyqmdqemjevzpibcbsicmqjiatxsclohuygdjsvbolgmctze"
VLM_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
VLM_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

# 实验室安全要求映射 (Index -> Class ID)
# 0: No Drinking -> Check Class 0 (Drinking)
# 1: No Eating -> Check Class 1 (Eating)
# 2: Gloves -> Check Class 7 (No Gloves)
# 3: Goggles -> Check Class 11 (No googles)
# 4: Mask -> Check Class 10 (No Mask)
# 5: Lab Coat -> Check Class 9 (No Lab coat)
# 6: Head Mask -> Check Class 8 (No Head Mask)
SAFETY_REQUIREMENTS_MAP = {
    0: 0,   # No Drinking -> Warn on 0
    1: 1,   # No Eating -> Warn on 1
    2: 7,   # Gloves required -> Warn on 7
    3: 11,  # Goggles required -> Warn on 11
    4: 10,  # Mask required -> Warn on 10
    5: 9,   # Lab Coat required -> Warn on 9
    6: 8    # Head Mask required -> Warn on 8
}

# 默认所有防护都必须 (1 = Required, 0 = Not Required)
current_safety_vector = [1, 1, 1, 1, 1, 1, 1]

# 允许上传的文件类型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}

# 创建必要的目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 加载YOLOv8模型
print("正在加载YOLOv8模型...")
try:
    model = YOLO('models/lab_safety_detection6/weights/best.pt')
    print("模型加载成功!")
    print(f"模型类别: {model.names}")
except Exception as e:
    print(f"模型加载失败: {e}")
    model = None

# 全局变量用于摄像头流
camera_active = False
camera_thread = None
frame_lock = threading.Lock()
current_frame = None
detection_stats = {"objects": {}, "fps": 0}
# 累计统计 (Session Total)
session_counts = {
    'Drinking': 0, 'Eating': 0, 'Gloves': 0, 'Googles': 0, 
    'Head Mask': 0, 'Lab Coat': 0, 'Mask': 0, 
    'No Gloves': 0, 'No Head Mask': 0, 'No Lab coat': 0, 
    'No Mask': 0, 'No googles': 0
}
# 存储上传视频的会话信息（每个上传视频一个会话）
video_sessions = {}

# 存储最新的 VLM 验证结果供前端展示
latest_vlm_result = {}
vlm_result_lock = threading.Lock()

# 需要触发警告的类别 id
ALERT_CLASS_IDS = {0, 1, 7, 8, 9, 10, 11}

# 全局服务器设置（可以通过 /set_settings 修改）
server_settings = {
    'conf_threshold': 0.3,
    'show_labels': True,
    'alerts_enabled': True,
    'json_logging_enabled': False,
    'language': 'zh'
}

# SQLite DB 路径
DB_PATH = os.path.join(app.config['UPLOAD_FOLDER'], 'detections.db')


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        # allow cross-thread access when needed
        db = g._database = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.row_factory = sqlite3.Row
    return db


def init_db():
    # 保留上传目录创建，但不再创建或初始化 SQLite 数据库。
    # 本应用改为使用 JSON 日志（static/uploads/detections.json），因此不需要数据库文件。
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    return


def save_warnings_json(source, warnings_list):
    if not server_settings.get('json_logging_enabled', False):
        return
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        logfile = os.path.join(app.config['UPLOAD_FOLDER'], 'detections.json')
        data = []
        if os.path.exists(logfile):
            with open(logfile, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = []
        for w in warnings_list:
            entry = {
                'source': source,
                'class_id': w.get('id'),
                'class_name': w.get('name'),
                'frame': w.get('frame'),
                'timestamp': w.get('timestamp')
            }
            data.append(entry)
        with open(logfile, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def save_warnings_per_video(filename, warnings_list):
    """为单个上传视频保存/追加日志文件 detections_<filename>.json"""
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        safe_name = f"detections_{filename}.json"
        logfile = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        data = []
        if os.path.exists(logfile):
            with open(logfile, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = []

        for w in warnings_list:
            entry = {
                'source': filename,
                'class_id': w.get('id'),
                'class_name': w.get('name'),
                'frame': w.get('frame'),
                'timestamp': w.get('timestamp')
            }
            data.append(entry)

        with open(logfile, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def call_vlm(image_base64, prompt):
    """调用 VLM 模型进行分析"""
    headers = {
        "Authorization": f"Bearer {VLM_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "stream": False
    }
    try:
        response = requests.post(VLM_API_URL, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"VLM Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"VLM Exception: {e}")
        return None

def process_frame(frame, conf_threshold=0.5):
    """处理单帧图像"""
    global latest_vlm_result
    if model is None:
        return frame, {}
    
    # 调整图像大小以提高处理速度
    height, width = frame.shape[:2]
    max_dim = 640
    scale = min(max_dim/height, max_dim/width, 1)
    
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame_resized = cv2.resize(frame, (new_width, new_height))
    else:
        frame_resized = frame
    
    # 进行预测
    results = model(frame_resized, conf=conf_threshold, verbose=False)
    
    # 统计检测结果
    stats = {"total": 0, "by_class": {}, "class_ids": []}
    
    # 计算当前不需要报警的类别ID
    ignored_ids = set()
    for idx, required in enumerate(current_safety_vector):
        if required == 0: # Not required
            target_class = SAFETY_REQUIREMENTS_MAP.get(idx)
            if target_class is not None:
                ignored_ids.add(target_class)

    if results and results[0].boxes is not None:
        result = results[0]
        boxes = result.boxes
        
        valid_indices = []
        
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names.get(cls_id, str(cls_id)) if hasattr(model, 'names') else str(cls_id)
            
            # 1. 检查是否在忽略列表中
            if cls_id in ignored_ids:
                continue
            
            # 2. 检查是否是 Drink/Eat (0/1) 并调用 VLM 验证 (Hybrid Strategy)
            if cls_id in [0, 1]:
                # 混合策略: Conf >= 0.7 -> Trust YOLO; 0.3 <= Conf < 0.7 -> Ask VLM; Conf < 0.3 -> Ignore
                if conf >= 0.7:
                    pass # High confidence, keep it
                elif conf < 0.3:
                    continue # Low confidence, skip
                else:
                    # 裁剪图像
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h, w = frame_resized.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        crop = frame_resized[y1:y2, x1:x2]
                        _, buffer = cv2.imencode('.jpg', crop)
                        crop_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        prompt = f"Is the person in this image {cls_name} (drinking or eating)? Please answer strictly with YES or NO. If unsure, answer YES."
                        print(f"Verifying {cls_name} with VLM (conf={conf:.2f})...")
                        # 注意：在实时流中调用 VLM 会导致显著延迟
                        vlm_resp = call_vlm(crop_b64, prompt)
                        print(f"VLM Response for {cls_name}: {vlm_resp}")
                        
                        # 保存 VLM 结果供前端展示
                        with vlm_result_lock:
                            latest_vlm_result = {
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "class_name": cls_name,
                                "image_base64": crop_b64,
                                "vlm_response": vlm_resp
                            }

                        if vlm_resp:
                            # 如果明确说 NO，则认为是误报
                            if "NO" in vlm_resp.upper() and "YES" not in vlm_resp.upper():
                                continue
            
            # 更新统计
            if cls_name not in stats["by_class"]:
                stats["by_class"][cls_name] = 0
            stats["by_class"][cls_name] += 1
            stats["class_ids"].append(cls_id)
            valid_indices.append(i)
        
        # 只绘制有效的框
        if len(valid_indices) < len(boxes):
             result.boxes = result.boxes[valid_indices]
        
        # 绘制检测结果
        frame_resized = result.plot()
    
    # 将处理后的图像恢复原始大小
    if scale < 1:
        frame = cv2.resize(frame_resized, (width, height))
    else:
        frame = frame_resized
    
    return frame, stats


def video_stream_generator(video_path, session_key, target_fps=30, conf_threshold=0.3):
    """按帧读取视频，进行检测，生成 MJPEG 流，并更新时间戳与会话统计。"""
    global video_sessions

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps_counter = 0
    fps_timer = time.time()
    fps = 0
    frame_index = 0

    while True:
        start_t = time.perf_counter()
        success, frame = cap.read()
        if not success:
            break

        processed_frame, stats = process_frame(frame, conf_threshold=conf_threshold)

        # 更新会话统计
        frame_index += 1
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        session = video_sessions.setdefault(session_key, {})
        session['fps'] = fps
        session['total'] = stats.get('total', 0)
        session['by_class'] = stats.get('by_class', {})
        session['frame_index'] = frame_index

        # 检查是否出现警告类
        warnings = []
        detected_ids = set(stats.get('class_ids', []))
        for cid in detected_ids:
            if cid in ALERT_CLASS_IDS:
                name = model.names.get(cid, str(cid)) if model else str(cid)
                warnings.append({"id": cid, "name": name, "frame": frame_index, "timestamp": datetime.now().isoformat()})

        # 如果找到警告，保存到会话并追加到 JSON 日志（全局日志 + 单视频日志）
        if warnings:
            session.setdefault('warnings', []).extend(warnings)
            try:
                save_warnings_json(session_key, warnings)
            except Exception:
                pass
            try:
                # 也保存为单独视频日志，便于下载
                save_warnings_per_video(session_key, warnings)
            except Exception:
                pass

        # 编码并发送帧
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # 控制帧率，尽量接近 target_fps
        elapsed = time.perf_counter() - start_t
        sleep_time = max(0, (1.0 / target_fps) - elapsed)
        time.sleep(sleep_time)

    cap.release()

def generate_frames():
    """生成摄像头视频流"""
    global camera_active, current_frame, detection_stats, session_counts
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    fps_counter = 0
    fps_timer = time.time()
    fps = 0
    frame_index = 0
    
    while camera_active:
        success, frame = cap.read()
        if not success:
            break
        
        # 处理帧（使用服务器设置的置信度）
        processed_frame, stats = process_frame(frame, conf_threshold=server_settings.get('conf_threshold', 0.3))
        
        # 更新累计统计
        for cls_name, count in stats.get('by_class', {}).items():
            if cls_name in session_counts:
                session_counts[cls_name] += count
            else:
                session_counts[cls_name] = session_counts.get(cls_name, 0) + count

        # 更新统计信息
        frame_index += 1
        detection_stats = stats
        detection_stats['frame_index'] = frame_index
        detection_stats['session_counts'] = session_counts # Add to response
        
        # 计算FPS
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_timer = time.time()
        
        detection_stats["fps"] = fps

        # 检查是否出现警告类并写入日志
        warnings = []
        detected_ids = set(stats.get('class_ids', []))
        for cid in detected_ids:
            if cid in ALERT_CLASS_IDS:
                name = model.names.get(cid, str(cid)) if model else str(cid)
                warnings.append({"id": cid, "name": name, "frame": frame_index, "timestamp": datetime.now().isoformat()})

        if warnings:
            detection_stats.setdefault('warnings', []).extend(warnings)
            try:
                save_warnings_json('camera', warnings)
            except Exception:
                pass
        
        # 添加FPS和统计信息到图像
        cv2.putText(processed_frame, f"FPS: {fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示检测数量
        cv2.putText(processed_frame, f"检测数量: {stats.get('total', 0)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 存储当前帧
        with frame_lock:
            current_frame = processed_frame
        
        # 编码为JPEG格式
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        # 生成视频流
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    """启动摄像头"""
    global camera_active, camera_thread, session_counts
    
    # 重置累计统计
    session_counts = {
        'Drinking': 0, 'Eating': 0, 'Gloves': 0, 'Googles': 0, 
        'Head Mask': 0, 'Lab Coat': 0, 'Mask': 0, 
        'No Gloves': 0, 'No Head Mask': 0, 'No Lab coat': 0, 
        'No Mask': 0, 'No googles': 0
    }
    
    if not camera_active:
        camera_active = True
        camera_thread = threading.Thread(target=generate_frames)
        camera_thread.daemon = True
        camera_thread.start()
    
    return jsonify({"status": "camera_started"})

@app.route('/stop_camera')
def stop_camera():
    """停止摄像头"""
    global camera_active
    camera_active = False
    return jsonify({"status": "camera_stopped"})

@app.route('/get_detection_stats')
def get_detection_stats():
    """获取检测统计信息"""
    # Create a copy to return
    stats = detection_stats.copy()
    
    # Append latest VLM result if exists
    with vlm_result_lock:
        if latest_vlm_result:
            stats['vlm_result'] = latest_vlm_result
            
    return jsonify(stats)

@app.route('/capture_frame')
def capture_frame():
    """捕获当前帧"""
    with frame_lock:
        if current_frame is not None:
            # 保存图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            cv2.imwrite(filepath, current_frame)
            
            # 返回图像数据
            _, buffer = cv2.imencode('.jpg', current_frame)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                "success": True,
                "filename": filename,
                "image_data": f"data:image/jpeg;base64,{img_str}"
            })
    
    return jsonify({"success": False, "message": "没有可用的帧"})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """上传并处理图像"""
    if 'file' not in request.files:
        return jsonify({"error": "没有文件"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    
    if file and allowed_file(file.filename):
        # 读取图像
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "无法读取图像"}), 400
        
        # 处理图像（使用服务器设置的置信度）
        processed_img, stats = process_frame(img, conf_threshold=server_settings.get('conf_threshold', 0.3))
        
        # 保存处理后的图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, processed_img)
        
        # 转换为base64
        _, buffer = cv2.imencode('.jpg', processed_img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # 生成检测统计HTML
        stats_html = f"<h5>检测统计</h5><p>总检测数: {stats.get('total', 0)}</p><ul>"
        for class_name, count in stats.get('by_class', {}).items():
            stats_html += f"<li>{class_name}: {count}</li>"
        stats_html += "</ul>"
        # 生成警告列表（若检测到需警告的类别）
        warnings = []
        for cid in set(stats.get('class_ids', [])):
            if cid in ALERT_CLASS_IDS:
                name = model.names.get(cid, str(cid)) if model else str(cid)
                warnings.append({"id": cid, "name": name, "timestamp": datetime.now().isoformat()})

        # 将警告追加到本地 JSON 日志（按图像文件名）
        if warnings:
            try:
                save_warnings_json(filename, warnings)
            except Exception:
                pass

        return jsonify({
            "success": True,
            "image_data": f"data:image/jpeg;base64,{img_str}",
            "stats": stats_html,
            "filename": filename,
            "warnings": warnings
        })
    
    return jsonify({"error": "不支持的文件类型"}), 400

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """上传并处理视频"""
    if 'file' not in request.files:
        return jsonify({"error": "没有文件"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    
    if file and allowed_file(file.filename):
        # 保存上传的视频
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, ext = os.path.splitext(filename)
        saved_filename = f"{base_name}_{timestamp}{ext}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        
        file.save(video_path)
        # 创建关联的空日志文件（detections_<saved_filename>.json），以便前端下载
        try:
            per_log = os.path.join(app.config['UPLOAD_FOLDER'], f"detections_{saved_filename}.json")
            if not os.path.exists(per_log):
                with open(per_log, 'w', encoding='utf-8') as f:
                    json.dump([], f)
        except Exception:
            pass
        # 在会话字典中注册该视频的日志文件名，便于 video_stats 返回和下载
        try:
            video_sessions[saved_filename] = {'logfile': f"detections_{saved_filename}.json"}
        except Exception:
            pass
        
        # 返回流播放的地址（stream 将在 /video_stream/<saved_filename> 提供）
        stream_url = f"/video_stream/{saved_filename}"
        return jsonify({
            "success": True,
            "message": "视频已上传，准备流式处理",
            "filename": saved_filename,
            "stream_url": stream_url
        })
    
    return jsonify({"error": "不支持的文件类型"}), 400

@app.route('/download/<filename>')
def download_file(filename):
    """下载处理后的文件"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename),
                     as_attachment=True)

@app.route('/model_info')
def model_info():
    """获取模型信息"""
    if model:
        return jsonify({
            "names": model.names,
            "num_classes": len(model.names),
            "input_size": model.overrides.get('imgsz', 640)
        })
    return jsonify({"error": "模型未加载"}), 400


@app.route('/video_stream/<filename>')
def video_stream(filename):
    """为上传的视频提供 MJPEG 流。"""
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "视频未找到"}), 404

    # 使用生成器按 30 FPS 流式返回处理后的视频帧（使用服务器设置的置信度）
    return Response(video_stream_generator(video_path, filename, target_fps=30, conf_threshold=server_settings.get('conf_threshold', 0.3)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_stats/<filename>')
def video_stats(filename):
    """返回上传视频当前的检测统计与警告记录（用于前端轮询）。"""
    session = video_sessions.get(filename, {})
    # 仅返回可 JSON 序列化的字段
    return jsonify({
        "fps": session.get('fps', 0),
        "total": session.get('total', 0),
        "by_class": session.get('by_class', {}),
        "frame_index": session.get('frame_index', 0),
        "warnings": session.get('warnings', []),
        "logfile": session.get('logfile', None)
    })


@app.route('/set_settings', methods=['POST'])
def set_settings():
    data = request.get_json() or {}
    conf = data.get('conf_threshold')
    show_labels = data.get('show_labels')
    alerts_enabled = data.get('alerts_enabled')
    language = data.get('language')
    if conf is not None:
        try:
            server_settings['conf_threshold'] = float(conf)
        except Exception:
            pass
    if show_labels is not None:
        server_settings['show_labels'] = bool(show_labels)
    if alerts_enabled is not None:
        server_settings['alerts_enabled'] = bool(alerts_enabled)
    if language in ('zh', 'en'):
        server_settings['language'] = language
    return jsonify({'status': 'ok', 'settings': server_settings})


@app.route('/get_settings')
def get_settings():
    return jsonify(server_settings)


@app.route('/history')
def history():
    """返回最近的告警历史，按时间降序"""
    try:
        logfile = os.path.join(app.config['UPLOAD_FOLDER'], 'detections.json')
        results = []
        if os.path.exists(logfile):
            with open(logfile, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    # 按时间倒序，取最近 500 条
                    data_sorted = sorted(data, key=lambda x: x.get('timestamp', ''), reverse=True)
                    results = data_sorted[:500]
                except Exception:
                    results = []

        return jsonify({'success': True, 'rows': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/stop_video_session/<filename>', methods=['POST'])
def stop_video_session(filename):
    """前端结束流式播放时可以调用此接口，清理服务器内存中的会话状态。"""
    if filename in video_sessions:
        try:
            video_sessions.pop(filename, None)
            # 可选：删除日志文件
            log_path = os.path.join(app.config['UPLOAD_FOLDER'], f"detections_{filename}.json")
            # 不删除日志，保留为记录；如果需要删除可取消注释下面行
            # if os.path.exists(log_path):
            #     os.remove(log_path)
        except Exception:
            pass
    return jsonify({"status": "stopped"})

@app.route('/analyze_scene', methods=['POST'])
def analyze_scene():
    """分析实验室场景并返回安全要求向量"""
    global current_safety_vector
    
    if 'file' not in request.files:
        return jsonify({"error": "没有文件"}), 400
    
    file = request.files['file']
    description = request.form.get('description', '')
    
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    
    if file and allowed_file(file.filename):
        img_bytes = file.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        prompt = (
            f"User description: {description}\n"
            "Analyze this lab image. Identify the lab category and scene. "
            "Based on safety standards, output a JSON array of 7 integers (0 or 1) representing requirements for:\n"
            "1. No Drinking (Drinking forbidden)\n"
            "2. No Eating (Eating forbidden)\n"
            "3. Gloves required\n"
            "4. Goggles required\n"
            "5. Mask required\n"
            "6. Lab Coat required\n"
            "7. Head Mask required\n"
            "1 means required/forbidden (check enabled), 0 means not required (ignore).\n"
            "Example output: [1, 1, 1, 1, 0, 1, 0]\n"
            "Only return the JSON array."
        )
        
        print("Calling VLM for scene analysis...")
        vlm_resp = call_vlm(img_b64, prompt)
        print(f"VLM Analysis Result: {vlm_resp}")
        
        vector = [1, 1, 1, 1, 1, 1, 1] # Default
        if vlm_resp:
            try:
                # 尝试从响应中提取数组
                start = vlm_resp.find('[')
                end = vlm_resp.find(']') + 1
                if start != -1 and end != -1:
                    json_str = vlm_resp[start:end]
                    vector = json.loads(json_str)
                    # Ensure length is 7
                    if len(vector) != 7:
                         vector = [1, 1, 1, 1, 1, 1, 1]
            except Exception as e:
                print(f"Error parsing VLM response: {e}")
        
        # Update global vector
        current_safety_vector = vector
        
        return jsonify({
            "success": True,
            "vector": vector,
            "raw_response": vlm_resp
        })
    
    return jsonify({"error": "Upload failed"}), 400

@app.route('/update_requirements', methods=['POST'])
def update_requirements():
    """更新安全要求向量"""
    global current_safety_vector
    data = request.get_json()
    vector = data.get('vector')
    if vector and isinstance(vector, list) and len(vector) == 7:
        current_safety_vector = vector
        return jsonify({"success": True, "vector": current_safety_vector})
    return jsonify({"error": "Invalid vector"}), 400

@app.route('/get_requirements')
def get_requirements():
    """获取当前安全要求向量"""
    return jsonify({"vector": current_safety_vector})

if __name__ == '__main__':
    # 确保模型目录存在
    os.makedirs('models/lab_safety_detection6/weights', exist_ok=True)
    
    # 注意：请将您的best.pt模型文件复制到上述目录中
    
    print("启动Flask应用...")
    print("访问地址: http://127.0.0.1:5000")
    # 初始化数据库
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
