# https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb

import os
import cv2
import glob
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import Sequence


RAW_DATA_PATH = "FoodSeg103/"
DATASET_PATH = "dataset/"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2


class Dataset:
    """Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
    """
    def __init__(self, images_dir, masks_dir, class_values=None, augmentation=None, preprocessing=None):
        self.images_ids = os.listdir(images_dir)
        self.images_ids.sort()
        self.masks_ids = os.listdir(masks_dir)
        self.masks_ids.sort()
        for (i, image) in enumerate(self.images_ids):
            image_id = os.path.basename(self.images_ids[i])
            mask_id = os.path.basename(self.masks_ids[i])
            assert image_id == mask_id
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.images_ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.masks_ids]
        for (i, image) in enumerate(self.images_fps):
            image_id = os.path.basename(self.images_fps[i])
            mask_id = os.path.basename(self.masks_fps[i])
            assert image_id == mask_id
        self.class_values = class_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask

    def __len__(self):
        return len(self.images_ids)


class Dataloader(Sequence):
    """Load data from dataset and form batches
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


def create_dataset():
    os.makedirs(DATASET_PATH)
    os.makedirs(DATASET_PATH + "train/img")
    os.makedirs(DATASET_PATH + "train/mask")
    os.makedirs(DATASET_PATH + "val/img")
    os.makedirs(DATASET_PATH + "val/mask")

    images = glob.glob(f"{RAW_DATA_PATH}img_dir/train/*")
    full_indices = range(len(images))
    train_indices = random.sample(full_indices, int(len(full_indices) * TRAIN_RATIO))
    val_indices = list(set(full_indices) - set(train_indices))
    print(len(full_indices), (len(train_indices) + len(val_indices)))

    for idx in train_indices:
        filename = os.path.basename(images[idx])
        shutil.copyfile(RAW_DATA_PATH + "img_dir/train/" + filename,
                        DATASET_PATH + "train/img/" + filename)
        shutil.copyfile(RAW_DATA_PATH + "ann_dir/train/" + filename,
                        DATASET_PATH + "train/mask/" + filename)

    for idx in val_indices:
        filename = os.path.basename(images[idx])
        shutil.copyfile(RAW_DATA_PATH + "img_dir/train/" + filename,
                        DATASET_PATH + "val/img/" + filename)
        shutil.copyfile(RAW_DATA_PATH + "ann_dir/train/" + filename,
                        DATASET_PATH + "val/mask/" + filename)

    shutil.copytree(RAW_DATA_PATH + "img_dir/test", DATASET_PATH + "test/img")
    shutil.copytree(RAW_DATA_PATH + "ann_dir/test", DATASET_PATH + "test/mask")


def change_channel():
    dataset_dirs = ["train", "val", "test"]
    for dataset_dir in dataset_dirs:
        shutil.copytree(f"{DATASET_PATH}{dataset_dir}/mask", f"{DATASET_PATH}{dataset_dir}/tmp")
        shutil.rmtree(f"{DATASET_PATH}{dataset_dir}/mask")
        os.makedirs(f"{DATASET_PATH}{dataset_dir}/mask")
        masks = glob.glob(f"{DATASET_PATH}{dataset_dir}/tmp/*")
        for mask in masks:
            m = cv2.imread(mask)
            r = m[:, :, 2]
            m[:, :, 0] = r
            m[:, :, 1] = r
            cv2.imwrite(f"{DATASET_PATH}{dataset_dir}/mask/{os.path.basename(mask)}", m)
        shutil.rmtree(f"{DATASET_PATH}{dataset_dir}/tmp")


def check_size():
    dataset_dirs = ["train", "val", "test"]
    for dataset_dir in dataset_dirs:
        images = glob.glob(f"{DATASET_PATH}{dataset_dir}/img/*")
        masks = glob.glob(f"{DATASET_PATH}{dataset_dir}/mask/*")
        for image in images:
            mask = [mask for mask in masks if os.path.basename(image) == os.path.basename(mask)][0]
            img = cv2.imread(image)
            img_height, img_width, img_channels = img.shape
            m = cv2.imread(mask)
            m_height, m_width, m_channels = m.shape
            if img_height != m_height or img_width != m_width:
                print("img: " + image + ", mask: " + mask)


def convert_jpg_to_png():
    dataset_dirs = ["train", "test"]
    for dataset_dir in dataset_dirs:
        shutil.copytree(f"{RAW_DATA_PATH}img_dir/{dataset_dir}", f"{RAW_DATA_PATH}img_dir/tmp")
        shutil.rmtree(f"{RAW_DATA_PATH}img_dir/{dataset_dir}")
        os.makedirs(f"{RAW_DATA_PATH}img_dir/{dataset_dir}")
        images = glob.glob(f"{RAW_DATA_PATH}img_dir/tmp/*")
        for image in images:
            jpg_img = cv2.imread(image)
            png_fp = os.path.join(f"{RAW_DATA_PATH}img_dir/{dataset_dir}",
                                  os.path.splitext(os.path.basename(image))[0] + ".png")
            cv2.imwrite(png_fp, jpg_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        shutil.rmtree(f"{RAW_DATA_PATH}img_dir/tmp")


def resize(width, height):
    dataset_dirs = ["train", "val", "test"]
    image_types = ["img", "mask"]
    for dataset_dir in dataset_dirs:
        for image_type in image_types:
            shutil.copytree(f"{DATASET_PATH}{dataset_dir}/{image_type}", f"{DATASET_PATH}{dataset_dir}/tmp")
            shutil.rmtree(f"{DATASET_PATH}{dataset_dir}/{image_type}")
            os.makedirs(f"{DATASET_PATH}{dataset_dir}/{image_type}")
            images = glob.glob(f"{DATASET_PATH}{dataset_dir}/tmp/*")
            for image in images:
                m = cv2.imread(image)
                m = cv2.resize(m, (width, height))
                cv2.imwrite(f"{DATASET_PATH}{dataset_dir}/{image_type}/{os.path.basename(image)}", m)
            shutil.rmtree(f"{DATASET_PATH}{dataset_dir}/tmp")


def show(img_path, mask_path):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    img = plt.imread(img_path)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    img = plt.imread(mask_path)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


if __name__ == '__main__':
    if not os.path.isdir(DATASET_PATH):
        convert_jpg_to_png()
        create_dataset()
        # change_channel()
        check_size()
        resize(480, 320)
