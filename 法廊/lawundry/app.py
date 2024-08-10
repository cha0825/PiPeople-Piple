from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from werkzeug.utils import secure_filename
import subprocess
import logging

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
FILENAMES = ["", ""] # filename 永遠只有2個

# Configure logging 就是那些原本在terminal 的東西會跑到app.log 這樣比較好真錯我覺得
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

@app.route('/')
def home():
    run_before_request()
    return render_template('about.html')
@app.route('/about')
def about():
    run_before_request()
    return render_template('about.html')
@app.route('/index')
def index():
    run_before_request()
    return render_template('index.html')

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)

# run BeforeRequest.sh 他的目的是為了要清空之前的資料 不然電腦早晚會爆炸
def run_before_request():
    try:
        result = subprocess.run(['./BeforeRequest.sh', 'arg1', 'arg2'], capture_output=True, text=True, check=True)
        print("run before request")
    except subprocess.CalledProcessError as e:
        print(f"Error executing script: {e.stderr}")

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    file = request.files['file']
    box_id = request.form['box_id']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # 根据 box_id 更新 FILENAMES
        if box_id == 'image-box1':
            FILENAMES[0] = filename
        elif box_id == 'image-box2':
            FILENAMES[1] = filename
        else:
            return jsonify({"success": False, "error": "Invalid box_id"}), 400

        with open('filenames.txt', 'w') as f:
            for name in FILENAMES:
                f.write(f"{name}\n")
        return jsonify({"success": True, "url": f"/uploads/{filename}"}), 201
    else:
        return jsonify({"success": False, "error": "File type not allowed"}), 400

# 把 upload 的 pic 丟到 uploads file 
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# run .sh file
@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        result = subprocess.run(['./runOverall.sh', 'arg1', 'arg2'], capture_output=True, text=True, check=True)

        with open('HSVresult.txt', 'r') as hsv_file:
            hsv = hsv_file.read().strip()
            print("HSV result:", hsv)
        with open('SSIMresult.txt', 'r') as ssim_file:
            ssim = ssim_file.read().strip()
            print("SSIM result:", ssim)
        
        return jsonify({"success": True, "ssim": ssim, "hsv": hsv}), 200
    
    except subprocess.CalledProcessError as e:
        return jsonify({"success": False, "error": e.stderr}), 500
    
# 讀 yolo特徵然後回傳給前端
@app.route('/get_features', methods=['POST'])
def get_features():
    print("run get features")
    features1 = []
    features2 = []

    filenames = ['YOLOresult1.txt', 'YOLOresult2.txt']
    for index, filename in enumerate(filenames):
        # 如果文件不存在就404
        if not os.path.isfile(filename):
            return jsonify({'error': f'File {filename} not found'}), 404

        try:
            with open(filename, 'r') as file:
                lines = file.readlines()
                print(f"Lines from {filename}: {lines}")
                num_features = int(lines[0].strip())
                features = [lines[i].strip() for i in range(1, num_features + 1)]
                if index == 0:
                    features1 = features
                else:
                    features2 = features
        except Exception as e:
            return jsonify({'error': f'Error reading file {filename}: {str(e)}'}), 500
    print("Features1:", features1)
    print("Features2:", features2)
    return jsonify({'features1': features1, 'features2': features2})

# 存被勾選的檔案是啥
@app.route('/save_image_src', methods=['POST'])
def save_image_src():
    data = request.json
    src1 = data.get('src1', '')
    src2 = data.get('src2', '')

    # 只保留 "/partpic/sharp_1_cake.jpg" 部分
    src1_path = src1.split('static')[-1]
    src2_path = src2.split('static')[-1]

    try:
        with open('partial-imagesChecked.txt', 'w') as file:
            file.write(f"{src1_path}\n")
            file.write(f"{src2_path}\n")
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# run runpartial.sh
@app.route('/run_partial_script', methods=['POST'])
def run_partial_script():
    try:
        # 使用 subprocess 运行 runPartial.sh
        result = subprocess.run(['./runPartial.sh'], capture_output=True, text=True, check=True)
        print("runPartial.sh executed successfully")
        
        with open('PartialHSVresult.txt', 'r') as hsv_file:
            hsv = hsv_file.read().strip()
            print("HSV result:", hsv)
        with open('PartialSSIMresult.txt', 'r') as ssim_file:
            ssim = ssim_file.read().strip()
            print("SSIM result:", ssim)
        
        return jsonify({"success": True, "ssim": ssim, "hsv": hsv}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # run
    app.run(debug=True)
