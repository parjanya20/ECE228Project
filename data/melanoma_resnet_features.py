import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.image as mpimg

data_dir = '/home/parjanya/UCSD_courses/ECE228/melanoma'
X = df['image_name'].values
X = [data_dir + '/train/' + x + '.jpg' for x in X]
print(len(X))

from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

resnet = models.resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        image = resnet(image)
    return image

batch_size = 100
num_batches = len(X) // batch_size
X_resnet = np.zeros((len(X), 512))
for i in range(num_batches):
    print(i)
    images = [load_image(x) for x in X[i*batch_size:(i+1)*batch_size]]
    images = torch.cat(images)
    images = images.view(images.size(0), -1)
    X_resnet[i*batch_size:(i+1)*batch_size] = images.numpy()

images = [load_image(x) for x in X[num_batches*batch_size:]]
images = torch.cat(images)
images = images.view(images.size(0), -1)
X_resnet[num_batches*batch_size:] = images.numpy()

np.save(data_dir + '/resnet_features.npy', X_resnet)



