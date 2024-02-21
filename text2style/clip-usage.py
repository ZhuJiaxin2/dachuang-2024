import pickle
import torch
import clip
from PIL import Image
from style_list import whole_style_list
import torch.nn.functional as F

device = "mps"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("./MUNIT_CLIP/hoovertower/hoovertowernight.jpg")).unsqueeze(0).to(device)

# tokens = clip.tokenize(whole_style_list)
# with open('whole_style_list_tokens.pkl', 'wb') as f:
#     pickle.dump(tokens, f)

with open('whole_style_list_tokens.pkl', 'rb') as f:
    text = pickle.load(f).to(device)

# image = preprocess(Image.open(hyperparameters['data_root'] + '/models')).unsqueeze(0).to(device)
# text = clip.tokenize(["starry night the painting"]).to(device)

#"starry night" Label probs: [[3.9341301e-01 6.0371584e-01 2.0429257e-03 5.9596459e-05 7.1085361e-04  5.7851514e-05]]
#"starry night the painting" Label probs: [[8.46049845e-01 1.53221369e-01 5.18488814e-04 1.51254135e-05  1.80412651e-04 1.46825514e-05]]

with torch.no_grad():
    # image_features = model.encode_image(image)
    # text_features = model.encode_text(text)
    # cos_sim = F.cosine_similarity(image_features, text_features, dim=-1).norm().cpu().numpy()
    # print(cos_sim)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image

print("Label probs:", probs)  