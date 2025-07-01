import os
import torch
from eva_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
import torch.nn.functional as F
import numpy as np

image_dir = "/mmu_mllm_hdd_2/yangzhuoran/dataset/laion-conceptual-captions-12m-webdataset/data/00000"
save_dir = "/mmu_mllm_hdd_2/yangzhuoran/dataset/laion-conceptual-captions-12m-webdataset/embedding/image_embedding/00000"

os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "EVA02-CLIP-bigE-14-plus"
pretrained = "/mmu_mllm_hdd_2/yangzhuoran/model/EVA02_CLIP_E_psz14_plus_s9B/EVA02_CLIP_E_psz14_plus_s9B.pt"

model, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
tokenizer = get_tokenizer(model_name)
model = model.to(device).eval()

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_dir, filename)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)  # [1, 1024]
            # TODO: pooling -> normalize
            # image_features /= image_features.norm(dim=-1, keepdim=True)

            feat_map = image_features.view(1, 1, 32, 32)  # [1, 1, 32, 32]
            pooled = F.avg_pool2d(feat_map, kernel_size=4, stride=4)  # [1, 1, 8, 8]
            final_feat = pooled.view(1, -1)  # [1, 64]
            final_feat /= final_feat.norm(dim=-1, keepdim=True)

        # 保存路径
        base_name = os.path.splitext(filename)[0]  # 去掉.jpg后缀
        save_path = os.path.join(save_dir, f"{base_name}.npy")
        np.save(save_path, final_feat.cpu().numpy())

        print(f"Saved: {save_path}, shape: {final_feat.shape}")
