#!/usr/bin/env python
# coding: utf-8

# In[15]:


import torch
from torchvision import transforms
from PIL import Image
import io
import matplotlib.pyplot as plt

model = torch.load('vks1.pt')
model.eval()


img= input('görselin dosya konumu giriniz:')

index = {1: "VKS1", 2: "VKS2", 3: "VKS3", 4: "VKS4", 5: "VKS5"}  
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(1),
        transforms.CenterCrop(200),
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    output = model(tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    conf, classes = torch.max(probs, 1)
    return conf.item(), index[classes.item()]

image_path = img
image = plt.imread(image_path)
plt.imshow(image)

with open(image_path, 'rb') as f:
    image_bytes = f.read()
    conf, y_pre = get_prediction(image_bytes=image_bytes)
    print("Tahmin edilen sınıf:", y_pre)


# In[ ]:





# In[ ]:




