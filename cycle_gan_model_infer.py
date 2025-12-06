import torch
import torchvision.transforms as T
from PIL import Image
from models import networks  # 引用你项目中的 models 文件夹

def load_cyclegan(checkpoint_path, direction="AtoB", device=None):
    """
    加载 CycleGAN 的生成器模型权重
    checkpoint_path: .pth 文件路径
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 定义生成器结构 (ResNet 9blocks)
    # 【核心修复】删掉了 gpu_ids 参数，因为你的底层函数不支持
    netG = networks.define_G(
        input_nc=3, 
        output_nc=3, 
        ngf=64, 
        netG='resnet_9blocks', 
        norm='instance', 
        use_dropout=False, 
        init_type='normal', 
        init_gain=0.02
        # 注意：这里删除了 gpu_ids=[]
    )
    
    # 2. 加载权重
    print(f"Loading model from {checkpoint_path}...")
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # 处理可能的 key 匹配问题
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
            
        # 加载参数
        netG.load_state_dict(state_dict)
        netG.to(device)
        netG.eval()  # 开启评估模式
        
        print(f"Model loaded successfully to {device}!")
        return netG, device
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def run_cyclegan(model, device, input_img_path, output_img_path, direction="AtoB"):
    """
    运行推理：读取图片 -> 预处理 -> 推理 -> 后处理 -> 保存
    """
    try:
        image = Image.open(input_img_path).convert('RGB')
        
        transforms = T.Compose([
            T.Resize((256, 256)), 
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        input_tensor = transforms(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 直接调用 model
            output_tensor = model(input_tensor)
        
        output_tensor = output_tensor.squeeze(0).cpu()
        output_tensor = (output_tensor + 1) / 2.0 * 255.0
        output_tensor = output_tensor.clamp(0, 255).type(torch.uint8)
        
        output_image = T.ToPILImage()(output_tensor)
        output_image.save(output_img_path)
        print(f"Result saved to {output_img_path}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise e