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

def list_objects_page(page):
    aws_files = []
    if 'Contents' in page:
        aws_files.extend([os.path.join(folder_path, os.path.basename(obj["Key"])) for obj in page['Contents'] if ".jpg" in obj['Key']])
    return aws_files

paginator = s3.get_paginator('list_objects_v2')
page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=folder_path)

aws_files = []
page_workers = 8
# Use ThreadPoolExecutor to parallelize listing
with concurrent.futures.ThreadPoolExecutor(max_workers=page_workers) as executor:
    # List objects in parallel
    futures = [executor.submit(list_objects_page, page) for page in tqdm(page_iterator, desc="Listing")]
    
    # Gather results from all futures
    for future in concurrent.futures.as_completed(futures):
        aws_files.extend(future.result())

with open('indexes/aws_file_list.pkl', 'wb') as f:
   pickle.dump(aws_files, f)

print(f'aws_files length: {len(aws_files)}')

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

def process_batch(batch_paths_images):
    paths, images = zip(*batch_paths_images)
    
    # Convert images to tensors and stack them
    images_tensor = torch.stack([transform(Image.fromarray(img)) for img in images]).to(device)
    
    with torch.no_grad():
        out_tensors = model(images_tensor)
    
    embeddings = [(path, out_tensor.cpu().numpy()) for path, out_tensor in zip(paths, out_tensors)]
    return embeddings

batch_size = 100
download_workers = 8
embeddings = []
representations = []

def download_and_process_batch(batch_paths):
    batch_images = []
    for path in batch_paths:
        response = s3.get_object(Bucket=bucket_name, Key=path)
        image_data = response['Body']
        img_np_array = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
        img = cv2.imdecode(img_np_array, cv2.IMREAD_COLOR)
        batch_images.append((path, img))
    return process_batch(batch_images)

batch_paths_list = [aws_files[i:i + batch_size] for i in range(0,len(aws_files), batch_size)]
embeddings = []

with concurrent.futures.ThreadPoolExecutor(max_workers=download_workers) as download_executor:
    for batch_paths in tqdm(batch_paths_list, desc="Downloading"):
        batch_futures = []
        for path in batch_paths:
            future = download_executor.submit(download_and_process_batch, [path])
            batch_futures.append(future)
        
        for future in concurrent.futures.as_completed(batch_futures):
            try:
                embeddings_list = future.result()
                embeddings.extend(embeddings_list)
            except Exception as e:
                print(f"An error occurred: {e}")

with open('indexes/aws_representations.pkl', 'wb') as f:
   pickle.dump(embeddings, f)