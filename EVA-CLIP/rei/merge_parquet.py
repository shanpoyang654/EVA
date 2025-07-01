from datasets import Dataset, Features, Image, Value, Array2D, Sequence
import os
import numpy as np
from PIL import Image as PILImage
from datasets import DatasetDict

image_dir = "/mmu_mllm_hdd_2/yangzhuoran/dataset/laion-conceptual-captions-12m-webdataset/data/00000"
text_dir = "/mmu_mllm_hdd_2/yangzhuoran/dataset/laion-conceptual-captions-12m-webdataset/data/00000"
embed_dir = "/mmu_mllm_hdd_2/yangzhuoran/dataset/laion-conceptual-captions-12m-webdataset/embedding/image_embedding/00000"





# 构建 rows
rows = []

sample_count = 0
max_samples = 5000  # 限制为前100个样本

print("max_samples:", max_samples)
for fname in os.listdir(embed_dir):
    if sample_count >= max_samples:
        break


    if fname.endswith(".npy"):
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(image_dir, base + ".jpg")
        txt_path = os.path.join(text_dir, base + ".txt")
        npy_path = os.path.join(embed_dir, fname)

        if not os.path.exists(img_path) or not os.path.exists(txt_path):
            continue

        with open(txt_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        image_embeds = np.load(npy_path).squeeze().astype(np.float32).tolist()

        rows.append({
            "image": img_path,        # 存储图像路径，datasets 会自动 load 成 image type
            "text": caption,
            "image_embeds": image_embeds,  # shape [64]
        })
        sample_count += 1

# 构建 dataset
features = Features({
    "image": Image(),
    "text": Value("string"),
    "image_embeds": Sequence(feature=Value("float32"), length=64)
})


ds = Dataset.from_list(rows, features=features)


# 构建DatasetDict而不仅是Dataset
ds_dict = DatasetDict({"train": ds})  # 用"train"作为默认split名称

# 保存为Parquet+元数据
save_dir = "/mmu_mllm_hdd_2/yangzhuoran/dataset/laion-conceptual-captions-12m-webdataset/parquet/00000_top5000"
ds_dict.save_to_disk(save_dir)  # 这会自动创建dataset_dict.json



print(f"Saved to {save_dir}")

