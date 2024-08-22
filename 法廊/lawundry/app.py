from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from werkzeug.utils import secure_filename
import subprocess
import re
from datetime import datetime
# import logging

app = Flask(__name__, template_folder='templates', static_folder='static')


# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('app.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
FILENAMES = ["", ""] # filename 永遠只有2個

@app.route('/')
def home():
    run_before_request()
    return render_template('index.html')
@app.route('/about')
def about():
    run_before_request()
    return render_template('about.html')
@app.route('/index')
def index():
    run_before_request()
    return render_template('index.html')
@app.route('/index/compare')
def comapre():
    image1_url = request.args.get('image1')
    image2_url = request.args.get('image2')
    return render_template('compare.html', image1=image1_url, image2=image2_url)
@app.route('/index/compare/introSSIM')
def introSSIM():
    return render_template('introSSIM.html')
@app.route('/index/compare/introHSV')
def introHSV():
    return render_template('introHSV.html')
@app.route('/index/compare/introCNN')
def introCNN():
    return render_template('introCNN.html')
@app.route('/index/compare/introRandomForest')
def introRandomForest():
    return render_template('introRandomForest.html')

# run BeforeRequest.sh 他的目的是為了要清空之前的資料 不然電腦早晚會爆炸
def run_before_request():
    try:
        result = subprocess.run(['./BeforeRequest.sh', 'arg1', 'arg2'], capture_output=True, text=True, check=True)
        print("run before request")
    except subprocess.CalledProcessError as e:
        print(f"Error executing script: {e.stderr}")

# 確認上傳文件存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def sanitize_filename(filename):
    filename = secure_filename(filename)
    # 對檔名做時間處理
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{timestamp}{ext}"
    return re.sub(r'\s+', '_', new_filename)

# uploads images
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    
    file = request.files['file']
    box_id = request.form['box_id']

    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = sanitize_filename(file.filename)
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

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

# run runOverall.sh file
@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        result = subprocess.run(['./runOverall.sh', 'arg1', 'arg2'], capture_output=True, text=True, check=True)

        random_dir = read_file('imagefileName.txt')
        hsv = read_file('HSVresult.txt')
        ssim = read_file('SSIMresult.txt')
        cnn = read_file('CNNresult.txt')
        
        return jsonify(success=True, random_dir=random_dir, ssim=ssim, hsv=hsv, cnn=cnn), 200
    
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
    src1_path = data.get('src1', '').split('static')[-1]
    src2_path = data.get('src2', '').split('static')[-1]

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
        
        result = subprocess.run(['./runPartial.sh'], capture_output=True, text=True, check=True)
        print(result.stdout)
        
        hsv = read_file('PartialHSVresult.txt')
        ssim = read_file('PartialSSIMresult.txt')
        
        return jsonify({"success": True, "ssim": ssim, "hsv": hsv}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # run
    app.run(debug=True)
