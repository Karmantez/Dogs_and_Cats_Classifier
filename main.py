#!/usr/bin/env python
# coding: utf-8

# In[28]:


# import testing
import resnet_model
import preprocessing
import importlib
from torchvision import transforms

import torch

# importlib.reload(testing)
importlib.reload(resnet_model)
importlib.reload(preprocessing)

print('Libraries are loaded successfully')


# In[29]:


# Loading data with PyTorch Tensor
# 1. Save all image sizes in the same size
# 2. Normalize the dataset with the mean and standard deviation of the dataset
# 3. Convert the image dataset to a PyTorch Tensor.

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

batch_train = preprocessing.batch_processing('/train', image_transform, 32, True, 3)
batch_valid = preprocessing.batch_processing('/valid', image_transform, 32, True, 3)


# In[30]:


model_ft = resnet_model.set_model_config()
setted_model = resnet_model.activate_model(model_ft, {'train' : batch_train, 'valid' : batch_valid}, epochs=10)


# In[ ]:




