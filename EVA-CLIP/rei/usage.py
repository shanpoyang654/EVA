import torch
from eva_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
import ipdb



model_name = "EVA02-CLIP-bigE-14-plus"  # EVA02_CLIP_E_psz14_plus_s9B
'''
['EVA01-CLIP-B-16', 'EVA01-CLIP-g-14', 'EVA01-CLIP-g-14-plus', 'EVA02-CLIP-B-16', 'EVA02-CLIP-bigE-14', 'EVA02-CLIP-bigE-14-plus', 'EVA02-CLIP-L-14', 'EVA02-CLIP-L-14-336'].
'''


# pretrained = "eva_clip" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"
pretrained = '/llm_reco_ssd/yangzhuoran/model/EVA02_CLIP_E_psz14_plus_s9B/EVA02_CLIP_E_psz14_plus_s9B.pt'

image_path = "/llm_reco_ssd/yangzhuoran/code/EVA/EVA-CLIP/cat2.png"
caption = ["a diagram", "a dog", "a cat"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
tokenizer = get_tokenizer(model_name)



model = model.to(device)

image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
print("image: ", image.shape) # torch.Size([1, 3, 224, 224])
text = tokenizer(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image) # torch.Size([1, 1024])
    ipdb.set_trace()
    print("image_features: ", image_features.shape)
    text_features = model.encode_text(text) # torch.Size([3, 1024])
    print("text_features: ", text_features.shape)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    print("image_features: ", image_features.shape)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    print("text_features: ", text_features.shape)
    
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[0.8275, 0.1372, 0.0352]]
'''
Label probs: tensor([[2.5357e-07, 9.9987e-01, 1.3135e-04]], device='cuda:0')
'''