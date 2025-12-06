import os
from flask import Flask, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename
import torch
from cycle_gan_model_infer import load_cyclegan, run_cyclegan

app = Flask(__name__)

# ==================== 1. 核心路径配置 (你的绝对路径) ====================

# 数据集根目录
DATASET_ROOT = r"E:\BaiduNetdiskDownload\ML_project\datasets\ink_painting"

# 模型文件夹路径
MODEL_DIR = r"E:\BaiduNetdiskDownload\ML_project\checkpoints\ink_shanshui_dual"

# 定义两个模型的具体路径
PATH_AtoB = os.path.join(MODEL_DIR, "latest_net_G_A.pth") # A -> B (照片 -> 水墨)
PATH_BtoA = os.path.join(MODEL_DIR, "latest_net_G_B.pth") # B -> A (水墨 -> 照片)

# ==================== 2. 静态资源配置 ====================
UPLOAD_FOLDER = os.path.join("static", "uploads")
RESULT_FOLDER = os.path.join("static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# ==================== 3. 加载模型 (双向) ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载 G_A (Photo -> Ink)
print(f"正在加载 [照片 -> 水墨] 模型...\n路径: {PATH_AtoB}")
try:
    model_photo_to_ink, _ = load_cyclegan(PATH_AtoB, direction="AtoB", device=device)
    print("✅ A->B 模型加载成功")
except Exception as e:
    print(f"❌ A->B 模型加载失败: {e}")
    model_photo_to_ink = None

# 加载 G_B (Ink -> Photo)
print(f"正在加载 [水墨 -> 照片] 模型...\n路径: {PATH_BtoA}")
try:
    model_ink_to_photo, _ = load_cyclegan(PATH_BtoA, direction="BtoA", device=device)
    print("✅ B->A 模型加载成功")
except Exception as e:
    print(f"❌ B->A 模型加载失败: {e}")
    model_ink_to_photo = None


def run_model(input_path: str, output_path: str, mode: str):
    """
    根据用户选择的 mode 调用不同的模型对象
    """
    if mode == "BtoA":
        if model_ink_to_photo is None:
            raise Exception("B->A 模型未成功加载")
        target_model = model_ink_to_photo
        print("执行推理: 水墨 -> 照片")
    else:
        if model_photo_to_ink is None:
            raise Exception("A->B 模型未成功加载")
        target_model = model_photo_to_ink
        print("执行推理: 照片 -> 水墨")
        
    # 执行推理
    run_cyclegan(
        target_model,
        device,
        input_img_path=input_path,
        output_img_path=output_path,
        direction="AtoB" # 参数已在 load 时固定，此处传参仅为占位
    )

# ==================== 4. 页面路由 ====================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    # POST：处理上传
    if "image" not in request.files:
        return "没有上传图片", 400
        
    # 获取用户选择的模式 (AtoB 或 BtoA)
    mode = request.form.get("mode", "AtoB") 

    file = request.files["image"]
    if file.filename == "":
        return "未选择文件", 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(input_path)

    # 结果文件名加前缀区分
    output_filename = f"{mode}_result_{filename}"
    output_path = os.path.join(app.config["RESULT_FOLDER"], output_filename)
    
    # 运行模型
    try:
        run_model(input_path, output_path, mode)
    except Exception as e:
        return f"生成失败: {str(e)}", 500

    original_url = url_for("static", filename=f"uploads/{filename}")
    processed_url = url_for("static", filename=f"results/{output_filename}")

    return render_template(
        "comparison.html",
        original_image=original_url,
        processed_image=processed_url,
    )

# ==================== 5. 数据集浏览路由 (修复 BuildError) ====================

@app.route("/dataset")
def dataset():
    """统计 trainA / trainB / testA / testB 里的图片数量"""
    datasets = []
    # 遍历标准文件夹
    for name in ["trainA", "trainB", "testA", "testB"]:
        img_dir = os.path.join(DATASET_ROOT, name)
        count = 0
        if os.path.isdir(img_dir):
            try:
                count = len([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
            except:
                count = 0
        datasets.append({"name": name, "count": count})
    return render_template("dataset.html", datasets=datasets)


@app.route("/dataset/<dataset_name>")
def dataset_detail(dataset_name):
    """展示某个数据集里的部分图片列表"""
    img_dir = os.path.join(DATASET_ROOT, dataset_name)
    images = []
    if os.path.isdir(img_dir):
        # 最多展示前 50 张
        try:
            for f in sorted(os.listdir(img_dir))[:50]:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    images.append({
                        "filename": f,
                        "url": url_for("dataset_file", dataset_name=dataset_name, filename=f)
                    })
        except:
            pass

    return render_template("dataset_detail.html", dataset_name=dataset_name, images=images)


@app.route("/dataset_files/<dataset_name>/<path:filename>")
def dataset_file(dataset_name, filename):
    """把数据集图片当作静态资源返回给前端"""
    img_dir = os.path.join(DATASET_ROOT, dataset_name)
    return send_from_directory(img_dir, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)