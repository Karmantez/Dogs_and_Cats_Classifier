#!/usr/bin/env python
# coding: utf-8

# In[4]:


from glob import glob
import random
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# In[5]:


# Where dogs and cats data is located
data_path = './dogs-vs-cats-redux-kernels-edition/train'
# total data * 0.1 = valid data set
verification_ratio = 0.1


# In[6]:


def create_directory():
    
    # Create directories ('train', 'valid') for learning and verification.
    if not os.path.exists(os.path.join(data_path, 'train')):
        os.mkdir(os.path.join(data_path, 'train'))
    
    if not os.path.exists(os.path.join(data_path, 'valid')):
        os.mkdir(os.path.join(data_path, 'valid'))
    
    # Create dogs and cats directory.
    for directory in ['train', 'valid']:
        for animal in ['dog', 'cat']:
            if not os.path.exists(os.path.join(data_path, directory, animal)):
                os.mkdir(os.path.join(data_path, directory, animal))


# In[7]:


def load_data():
    
    # Create a directory first before categorizing dogs and cats data.
    create_directory()
    
    # Read a list of all files in a folder.
    files = glob(os.path.join(data_path, '*.jpg'))
    
    # Shuffle the file list to increase reliability.
    random.shuffle(files)
    
    # Boundary value between training data and verification data
    boundary = (int)(len(files) * verification_ratio)
    
    # process train dataset
    for file in files[:boundary]:
        filenames = file.split('\\')[-1].split('.')
        os.rename(file, os.path.join(data_path, 'train', filenames[0], filenames[1]+'.'+filenames[2]))
        
    # process valid dataset
    for file in files[boundary:]:
        filenames = file.split('\\')[-1].split('.')
        os.rename(file, os.path.join(data_path, 'valid', filenames[0], filenames[1]+'.'+filenames[2]))


# In[8]:


def batch_processing(directory, image_transform, batch_size, shuffle, num_workers):
    images = ImageFolder(data_path + directory, image_transform)
    return DataLoader(images, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) 

