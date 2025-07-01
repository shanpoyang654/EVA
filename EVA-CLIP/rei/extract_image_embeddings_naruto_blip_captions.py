import os
import torch
from datasets import load_dataset
from eva_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 配置路径
save_dir = "/llm_reco_ssd/yangzhuoran/dataset/naruto-blip-captions/embeddings"
os.makedirs(save_dir, exist_ok=True)

# 加载数据集
dataset = load_dataset("lambdalabs/naruto-blip-captions", split="train")
print(f"Loaded dataset with {len(dataset)} samples")

# 初始化模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "EVA02-CLIP-bigE-14-plus"
pretrained = "/llm_reco_ssd/yangzhuoran/model/EVA02_CLIP_E_psz14_plus_s9B/EVA02_CLIP_E_psz14_plus_s9B.pt"

model, _, preprocess = create_model_and_transforms(
    model_name, 
    pretrained, 
    force_custom_clip=True
)
model = model.to(device).eval()

# 提取嵌入
for idx, sample in enumerate(tqdm(dataset)):
    try:
        # 处理图像
        image = sample["image"] # .convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # 提取特征
        with torch.no_grad(), torch.cuda.amp.autocast():
            # 获取全局特征 [1, 1024]
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 转换为空间特征并池化到64维
            feat_map = image_features.view(1, 1, 32, 32)  # 假设空间排列为32x32
            pooled = F.avg_pool2d(feat_map, kernel_size=4, stride=4)  # [1, 1, 8, 8]
            final_feat = pooled.view(1, -1)  # [1, 64]
        
        # 保存嵌入
        save_path = os.path.join(save_dir, f"{idx:06d}.npy")
        np.save(save_path, final_feat.cpu().numpy())
        
    except Exception as e:
        print(f"Error processing sample {idx}: {str(e)}")
        continue

print(f"Finished! Embeddings saved to {save_dir}")