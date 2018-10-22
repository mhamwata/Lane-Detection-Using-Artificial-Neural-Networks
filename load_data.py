import cv2
import os
import random
import glob
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

def merge(*args):
    list = []
    while True:
        for arg in args:
            xy = next(arg)
            list.append(xy)
        yield tuple(list)

def load_batch(directory,batch_size,target_size):
    batch = []
    filenames = glob.glob(directory)
    print('Loading sample batch from ',directory)
    assert filenames != [], "Empty or wrong directory"
    for i in range(0,batch_size):
        filepath = random.choice(filenames)
        img = cv2.imread(filepath)
        img = cv2.resize(img,tuple(target_size[0:2]))
        batch.append(img)
    return np.array(batch)

def load(directory,data_gen_dict,batch_size,target_size,seed,train_mode=True):
    print('Loading images from ',directory)
    if train_mode:
        data_gen_dict = data_gen_dict
        #load sample batch of data
        sample_batch = load_batch(os.path.join(directory,'train_set\\data\\*.png'),batch_size,target_size)
    else:
        data_gen_dicts = dict()
        sample_batch = np.array([])

    train_data_gen = ImageDataGenerator(data_gen_dict)
    if (sample_batch.size != 0): train_data_gen.fit(sample_batch)
    train_generator = train_data_gen.flow_from_directory(
                                                   directory=os.path.join(directory,'train_set'),
                                                   target_size=target_size,
                                                   batch_size=batch_size,
                                                   seed=seed,
                                                   class_mode=None)
    
    gt_data_gen = ImageDataGenerator(data_gen_dict)

    gt_generator = gt_data_gen.flow_from_directory(
                                                   directory=os.path.join(directory,'gt_set'),
                                                   target_size=target_size,
                                                   batch_size=batch_size, 
                                                   seed=seed,
                                                   color_mode='grayscale',
                                                   class_mode=None)

    gen = merge(train_generator,gt_generator)

    return gen