# https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTModel ViTModel

from transformers import AutoImageProcessor, ViTModel
import torch
# from datasets import load_dataset
from PIL import Image

# image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
# model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
# model.save_pretrained("pretrainVIT")
# image_processor.save_pretrained("pretrainVIT")

# load model
model = ViTModel.from_pretrained("pretrainVIT")
image_processor = AutoImageProcessor.from_pretrained("pretrainVIT")
# load pic
image = Image.open("data3.png").convert("RGB")
inputs = image_processor(image, return_tensors="pt")
# model 
with torch.no_grad():
    outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
# pic info
# print(list(last_hidden_states.shape))
cls_embedding = last_hidden_states[:, 0, :]
print(cls_embedding)

#  ================ 分類 ================ 
# from transformers import AutoImageProcessor, ViTForImageClassification
# import torch
# from datasets import load_dataset
# from PIL import Image

# # dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
# # image = dataset["test"]["image"][0]
# image = Image.open("image015.png").convert("RGB")

# image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# inputs = image_processor(image, return_tensors="pt")

# with torch.no_grad():
#     logits = model(**inputs).logits

# # model predicts one of the 1000 ImageNet classes
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])