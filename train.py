# https://github.com/qubvel/segmentation_models
# https://github.com/qubvel/segmentation_models/issues/88
# https://github.com/qubvel/segmentation_models/issues/374
# https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb

import os
import numpy as np
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt
import segmentation_models as sm

from dataset import show
from dataset import Dataset
from dataset import visualize
from dataset import Dataloader
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from segmentation_models.losses import DiceLoss
from segmentation_models.losses import CategoricalFocalLoss
from segmentation_models.metrics import IOUScore
from segmentation_models.metrics import FScore
from segmentation_models import Linknet
from segmentation_models import PSPNet
from segmentation_models import Unet
from segmentation_models import FPN


CATEGORY_PATH = "dataset/category_id.txt"
TRAIN_PATH = "dataset/train/"
VAL_PATH = "dataset/val/"
TEST_PATH = "dataset/test/"

BATCH_SIZE = 8
MODEL = "efficientnetb3"
CHECKPOINT_PATH = "checkpoints/" + MODEL + ".h5"
FIGURE_PATH = "figures/" + MODEL + ".png"
MODEL_PATH = "models/" + MODEL + ".h5"
LOG_PATH = "logs/" + MODEL + ".log"


def get_classes():
    categories = pd.read_csv(CATEGORY_PATH, sep="\t", names=["id", "name"])
    ids = categories["id"].to_list()
    classes = categories["name"].to_list()
    return ids, classes


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=320, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf([A.CLAHE(p=1),
                 A.RandomBrightnessContrast(p=1),
                 A.RandomGamma(p=1)], p=0.9),
        A.OneOf([A.Sharpen(p=1),
                 A.Blur(blur_limit=3, p=1),
                 A.MotionBlur(blur_limit=3, p=1)], p=0.9),
        A.OneOf([A.RandomBrightnessContrast(p=1),
                 A.HueSaturationValue(p=1)], p=0.9),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_val_test_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [A.PadIfNeeded(384, 480)]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    _transform = [A.Lambda(image=preprocessing_fn)]
    return A.Compose(_transform)


def compile_model():
    dice_loss = DiceLoss(class_weights=np.append(np.ones(12), 0.2))
    focal_loss = CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    iou_score = IOUScore(threshold=0.5)
    f1_score = FScore(threshold=0.5)
    if not os.path.exists(CHECKPOINT_PATH):
        model = Unet(MODEL, classes=len(get_classes()[0]) + 1, activation="softmax", encoder_weights="imagenet")
        optimizer = Adam(learning_rate=0.00001)
        model.compile(loss=total_loss, optimizer=optimizer, metrics=[iou_score, f1_score])
    else:
        custom_objects = {
            "iou_score": iou_score,
            "f1-score": f1_score,
            "dice_loss_plus_1focal_loss": total_loss
        }
        model = load_model(CHECKPOINT_PATH, custom_objects=custom_objects)
        print("Checkpoint Model Loaded")
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor="val_loss", save_best_only=True,
                                 verbose=1, save_weights_only=False)
    lr = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=10)
    csv_logger = CSVLogger(LOG_PATH)
    print(model.summary())
    return model, early_stopping, checkpoint, lr, csv_logger


def train():
    preprocess_input = sm.get_preprocessing(MODEL)
    ids, classes = get_classes()
    train_dataset = Dataset(TRAIN_PATH + "img", TRAIN_PATH + "mask", class_values=ids,
                            augmentation=get_training_augmentation(),
                            preprocessing=get_preprocessing(preprocess_input))
    val_dataset = Dataset(VAL_PATH + "img", VAL_PATH + "mask", class_values=ids,
                          augmentation=get_val_test_augmentation(),
                          preprocessing=get_preprocessing(preprocess_input))
    test_dataset = Dataset(TEST_PATH + "img", TEST_PATH + "mask", class_values=ids,
                           augmentation=get_val_test_augmentation(),
                           preprocessing=get_preprocessing(preprocess_input))
    train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = Dataloader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)
    print(train_dataloader[0][0].shape)
    print(train_dataloader[0][1].shape)
    model, early_stopping, checkpoint, lr, csv_logger = compile_model()
    history = model.fit(train_dataloader,
                        steps_per_epoch=len(train_dataloader),
                        epochs=300,
                        callbacks=[early_stopping, checkpoint, lr, csv_logger],
                        validation_data=val_dataloader,
                        validation_steps=len(val_dataloader))

    train_score = model.evaluate_generator(train_dataloader)
    print("Train Loss: ", train_score[0])
    print("Train Mean IoU: ", train_score[1])
    print("Train Mean F1: ", train_score[2])

    val_score = model.evaluate_generator(val_dataloader)
    print("Validation Loss: ", val_score[0])
    print("Validation Mean IoU: ", val_score[1])
    print("Validation Mean F1: ", val_score[2])

    test_score = model.evaluate_generator(test_dataloader)
    print("Test Loss: ", test_score[0])
    print("Test Mean IoU: ", test_score[1])
    print("Test Mean F1: ", test_score[2])

    model.save(MODEL_PATH)

    plt.figure(figsize=(12, 8))
    plt.title("EVALUATION")
    plt.subplot(2, 2, 1)
    plt.plot(history.history["iou_score"], label="IoU")
    plt.plot(history.history["val_iou_score"], label="Val_IoU")
    plt.legend()
    plt.title("IoU Evaluation")
    plt.subplot(2, 2, 2)
    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["val_loss"], label="Val_Loss")
    plt.legend()
    plt.title("Loss Evaluation")
    # plt.show()
    plt.savefig(FIGURE_PATH)


if __name__ == '__main__':
    # show(TRAIN_PATH + "img/00000001.png", TRAIN_PATH + "mask/00000001.png")
    # dataset = Dataset(TRAIN_PATH + "img", TRAIN_PATH + "mask", class_values=get_classes()[0])
    # dataset = Dataset(TRAIN_PATH + "img", TRAIN_PATH + "mask", class_values=get_classes()[0],
    #                   augmentation=get_training_augmentation())
    # preprocess_input = sm.get_preprocessing(MODEL)
    # dataset = Dataset(TRAIN_PATH + "img", TRAIN_PATH + "mask", class_values=get_classes()[0],
    #                   augmentation=get_training_augmentation(),
    #                   preprocessing=get_preprocessing(preprocess_input))
    # image, mask = dataset[0]
    # visualize(image=image,
    #           sky_mask=mask[..., 0].squeeze(),
    #           car_mask=mask[..., 8].squeeze(),
    #           pedestrian_mask=mask[..., 9].squeeze(),
    #           unlabelled_mask=mask[..., 11].squeeze())
    train()
