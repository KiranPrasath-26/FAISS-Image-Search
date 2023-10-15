import os
import asyncio
import concurrent.futures
import torch
import cv2
import pickle
from io import BytesIO
from torchvision import models, transforms
from PIL import Image, ImageFile
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import faiss
import pandas as pd
import math
import matplotlib.pyplot as plt
from IPython.display import display
ImageFile.LOAD_TRUNCATED_IMAGES = True

import boto3
from botocore.exceptions import NoCredentialsError  

access_key = ''
secret_key = ''
bucket_name = ''
folder_path = ''

s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)

model.eval()
model.fc = nn.Identity()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def encode(image):
    input_tensor = transform(image).unsqueeze(0)

    if input_tensor.size()[1] == 3:
        with torch.no_grad():
            out_tensor = model(input_tensor)
        image.close()
        return out_tensor.numpy()
    else:
        image.close()
        return None

index = faiss.read_index("indexes/jewel_trained.index")

with open('indexes/aws_unique_rep.pkl', 'rb') as f:
   aws_rep = pickle.load(f)

path, embeddings = zip(*aws_rep)
path = list(path)
embeddings = list(embeddings)

emb = np.array(embeddings, dtype='float32')

response = s3.get_object(Bucket=bucket_name, Key=path[0]) ## change this part to get the target image from the app
image_data = response['Body']
img_np_array = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
target_img = cv2.imdecode(img_np_array, cv2.IMREAD_COLOR)
target_rep = encode(Image.fromarray(target_img))
print(path[0])
display(Image.fromarray(target_img).resize((300,300)))
faiss.normalize_L2(target_rep)


k = 20 ## top k results
D, I = index.search(target_rep, k)
I = I[0]
result = [path[i] for i in I]
result

## result is the final answer we need to send to the api