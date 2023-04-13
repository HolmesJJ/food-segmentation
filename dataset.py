# https://github.com/divamgupta/image-segmentation-keras

import os
import cv2
import glob
import random
import shutil


RAW_DATA_PATH = "UECFoodPixComplete/"
DATASET_PATH = "dataset/"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2


def create_dataset():
    os.makedirs(DATASET_PATH)
    os.makedirs(DATASET_PATH + "train/img")
    os.makedirs(DATASET_PATH + "train/mask")
    os.makedirs(DATASET_PATH + "train/mask_rgb")
    os.makedirs(DATASET_PATH + "val/img")
    os.makedirs(DATASET_PATH + "val/mask")
    os.makedirs(DATASET_PATH + "val/mask_rgb")
    os.makedirs(DATASET_PATH + "test/mask")

    images = glob.glob(f"{RAW_DATA_PATH}train/img/*")
    full_indices = range(len(images))
    train_indices = random.sample(full_indices, int(len(full_indices) * TRAIN_RATIO))
    val_indices = list(set(full_indices) - set(train_indices))
    print(len(full_indices), (len(train_indices) + len(val_indices)))

    for idx in train_indices:
        filename = os.path.splitext(os.path.basename(images[idx]))[0]
        shutil.copyfile(RAW_DATA_PATH + "train/img/" + filename + ".jpg",
                        DATASET_PATH + "train/img/" + filename + ".jpg")
        shutil.copyfile(RAW_DATA_PATH + "train/mask/" + filename + ".png",
                        DATASET_PATH + "train/mask_rgb/" + filename + ".png")

    for idx in val_indices:
        filename = os.path.splitext(os.path.basename(images[idx]))[0]
        shutil.copyfile(RAW_DATA_PATH + "train/img/" + filename + ".jpg",
                        DATASET_PATH + "val/img/" + filename + ".jpg")
        shutil.copyfile(RAW_DATA_PATH + "train/mask/" + filename + ".png",
                        DATASET_PATH + "val/mask_rgb/" + filename + ".png")

    shutil.copytree(RAW_DATA_PATH + "test/img", DATASET_PATH + "test/img")
    shutil.copytree(RAW_DATA_PATH + "test/mask", DATASET_PATH + "test/mask_rgb")


def change_channel():
    dataset_dirs = ["train", "val", "test"]
    for dataset_dir in dataset_dirs:
        masks = glob.glob(f"{DATASET_PATH}{dataset_dir}/mask_rgb/*")
        for mask in masks:
            m = cv2.imread(mask)
            m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{DATASET_PATH}{dataset_dir}/mask/{os.path.basename(mask)}", m)
        shutil.rmtree(f"{DATASET_PATH}{dataset_dir}/mask_rgb")


if __name__ == '__main__':
    if not os.path.isdir(DATASET_PATH):
        create_dataset()
        change_channel()
