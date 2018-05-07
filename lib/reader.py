import random
from paddle.v2.image import load_and_transform
import paddle.v2 as paddle
from multiprocessing import cpu_count
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2


def get_train_and_val_list(train_path, save_train_path='data/split/train.txt', save_val_path='data/split/val.txt'):
    all_pd = pd.read_csv(train_path, sep=" ", header=None, names=['ImageName', 'label'])
    train_pd, val_pd = train_test_split(all_pd, test_size=0.15, random_state=43, stratify=all_pd['label'])
    train_pd.to_csv(save_train_path, sep=" ", header=None, index=None)
    val_pd.to_csv(save_val_path, sep=" ", header=None, index=None)


def train_mapper(sample):
    '''
    map image path to type needed by model input layer for the training set
    '''
    img, label = sample
    img = paddle.image.load_image(img)
    img = paddle.image.simple_transform(img, 256, 224, True)
    return img.flatten().astype('float32'), label


def test_mapper(sample):
    '''
    map image path to type needed by model input layer for the test set
    '''
    img, label = sample
    img = paddle.image.load_image(img)
    img = paddle.image.simple_transform(img, 256, 224, True)
    return img.flatten().astype('float32'), label


def train_reader(train_list, train_dir='data/train', buffered_size=1024):
    def reader():
        with open(train_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split(' ')
                img_path = os.path.join(train_dir, img_path)
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(train_mapper, reader,
                                      cpu_count(), buffered_size)


def test_train_reader(train_list, train_dir='data/train', buffered_size=1024):
    with open(train_list, 'r') as f:
        lines = [line.strip() for line in f]
        for line in lines:
            img_path, lab = line.strip().split(' ')
            img_path = os.path.join(train_dir, img_path)
            print img_path, int(lab)


def test_reader(test_list, test_dir='data/train', buffered_size=1024):
    def reader():
        with open(test_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split(' ')
                img_path = os.path.join(test_dir, img_path)
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(test_mapper, reader,
                                      cpu_count(), buffered_size)


if __name__ == '__main__':
    # for im in train_reader('data/train.list'):
    #     print(len(im[0]))
    # for im in train_reader('data/test.list'):
    #     print(len(im[0]))
    get_train_and_val_list('data/train.txt')
    # test_train_reader('data/train.list')
