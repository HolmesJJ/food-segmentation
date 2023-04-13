# https://github.com/divamgupta/image-segmentation-keras
# https://github.com/divamgupta/image-segmentation-keras/issues/284
# https://blog.csdn.net/u012897374/article/details/80142744

import os
import shutil
import matplotlib.pyplot as plt

from keras.metrics import MeanIoU
from keras.metrics import Accuracy
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras_segmentation.models.fcn import fcn_32_resnet50
from imgaug import augmenters as iaa


TRAIN_PATH = "dataset/train/"
VAL_PATH = "dataset/val/"
TEST_PATH = "dataset/test/"
AUGMENTED_PATH = "augmented/"
AUGMENTED_TRAIN_PATH = "augmented/train/"
AUGMENTED_VAL_PATH = "augmented/val/"
AUGMENTED_TEST_PATH = "augmented/test/"

BATCH_SIZE = 32
NUM_CLASSES = 103
MODEL = "fcn_32_resnet50"
CHECKPOINT_PATH = "checkpoints/" + MODEL + ".h5"
FIGURE_PATH = "figures/" + MODEL + ".png"
MODEL_PATH = "models/" + MODEL + ".h5"
LOG_PATH = "logs/" + MODEL + ".log"
TMP_PATH = "tmp/" + MODEL


def show_food(name):
    original_image = TRAIN_PATH + "/img/" + name + ".jpg"
    label_image_semantic = TRAIN_PATH + "/mask/" + name + ".png"
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    img = plt.imread(original_image)
    plt.title(name)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    img = plt.imread(label_image_semantic)
    plt.title(name)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def run_data_augmentation():
    if os.path.isdir(AUGMENTED_PATH):
        shutil.rmtree(AUGMENTED_PATH)
    os.makedirs(AUGMENTED_PATH)
    os.makedirs(AUGMENTED_TRAIN_PATH)
    os.makedirs(AUGMENTED_TRAIN_PATH + "img/")
    os.makedirs(AUGMENTED_TRAIN_PATH + "mask/")
    os.makedirs(AUGMENTED_VAL_PATH)
    os.makedirs(AUGMENTED_VAL_PATH + "img/")
    os.makedirs(AUGMENTED_VAL_PATH + "mask/")
    os.makedirs(AUGMENTED_TEST_PATH)
    os.makedirs(AUGMENTED_TEST_PATH + "img/")
    os.makedirs(AUGMENTED_TEST_PATH + "mask/")
    datagen_args = dict(rescale=1 / 255,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        vertical_flip=True,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        rotation_range=20,
                        fill_mode="nearest")
    train_img_datagen = ImageDataGenerator(**datagen_args)
    train_mask_datagen = ImageDataGenerator(**datagen_args)
    val_test_datagen = ImageDataGenerator(rescale=1 / 255)

    seed = 42
    i = 0
    for batch in train_img_datagen.flow_from_directory(TRAIN_PATH,
                                                       classes=["img"],
                                                       save_to_dir=AUGMENTED_TRAIN_PATH + "img",
                                                       batch_size=BATCH_SIZE,
                                                       class_mode=None,
                                                       seed=seed):
        i += 1
    print("Train img augmented size: " + str(i))
    i = 0
    for batch in train_mask_datagen.flow_from_directory(TRAIN_PATH,
                                                        classes=["mask"],
                                                        save_to_dir=AUGMENTED_TRAIN_PATH + "mask",
                                                        batch_size=BATCH_SIZE,
                                                        class_mode=None,
                                                        seed=seed):
        i += 1
    print("Train mask augmented size: " + str(i))
    i = 0
    for batch in val_test_datagen.flow_from_directory(VAL_PATH,
                                                      classes=["img"],
                                                      save_to_dir=AUGMENTED_VAL_PATH + "img",
                                                      batch_size=BATCH_SIZE,
                                                      class_mode=None,
                                                      seed=seed):
        i += 1
    print("Validation img augmented size: " + str(i))
    for batch in val_test_datagen.flow_from_directory(VAL_PATH,
                                                      classes=["mask"],
                                                      save_to_dir=AUGMENTED_VAL_PATH + "mask",
                                                      batch_size=BATCH_SIZE,
                                                      class_mode=None,
                                                      seed=seed):
        i += 1
    print("Validation img augmented size: " + str(i))
    for batch in val_test_datagen.flow_from_directory(TEST_PATH,
                                                      classes=["img"],
                                                      save_to_dir=AUGMENTED_TEST_PATH + "img",
                                                      batch_size=BATCH_SIZE,
                                                      class_mode=None,
                                                      seed=seed):
        i += 1
    print("Test img augmented size: " + str(i))
    for batch in val_test_datagen.flow_from_directory(TEST_PATH,
                                                      classes=["mask"],
                                                      save_to_dir=AUGMENTED_TEST_PATH + "mask",
                                                      batch_size=BATCH_SIZE,
                                                      class_mode=None,
                                                      seed=seed):
        i += 1
    print("Test img augmented size: " + str(i))


def custom_augmentation():
    return iaa.Sequential([
            iaa.Fliplr(0.2),
            iaa.Flipud(0.2),
            iaa.Sometimes(0.2, iaa.Crop(percent=(0, 0.2))),
            iaa.Sometimes(0.2, iaa.LinearContrast((0.75, 1.5))),
            iaa.Sometimes(0.2, iaa.AverageBlur(k=(2, 7))),
            iaa.Sometimes(0.2, iaa.CoarseDropout(0.1, size_percent=0.1))
        ], random_order=True)


def compile_model():
    if not os.path.exists(CHECKPOINT_PATH):
        model = fcn_32_resnet50(n_classes=NUM_CLASSES)
        optimizer = Adam(learning_rate=0.00001)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=[
            Accuracy(), MeanIoU(num_classes=NUM_CLASSES)])
    else:
        model = load_model(CHECKPOINT_PATH)
        print("Checkpoint Model Loaded")
    early_stopping = EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor="val_accuracy", save_best_only=True,
                                 verbose=1, save_weights_only=False)
    lr = ReduceLROnPlateau(monitor="val_accuracy", mode="max", patience=10)
    csv_logger = CSVLogger(LOG_PATH)
    print(model.summary())
    return model, early_stopping, checkpoint, lr, csv_logger


def train():
    if not os.path.exists(TMP_PATH):
        os.makedirs(TMP_PATH)
    model, early_stopping, checkpoint, lr, csv_logger = compile_model()
    history = model.train(train_images=TRAIN_PATH + "img/",
                          train_annotations=TRAIN_PATH + "mask/",
                          val_images=VAL_PATH + "img/",
                          val_annotations=VAL_PATH + "mask/",
                          checkpoints_path=TMP_PATH,
                          optimizer_name="adam",
                          epochs=300,
                          do_augment=True,
                          custom_augmentation=custom_augmentation,
                          callbacks=[early_stopping, checkpoint, lr, csv_logger],
                          batch_size=BATCH_SIZE)

    print(("=" * 10) + " Train Evaluation " + ("=" * 10))
    print(model.evaluate_segmentation(inp_images_dir=TRAIN_PATH + "img/",
                                      annotations_dir=TRAIN_PATH + "mask/"))

    print(("=" * 10) + " Validation Evaluation " + ("=" * 10))
    print(model.evaluate_segmentation(inp_images_dir=VAL_PATH + "img/",
                                      annotations_dir=VAL_PATH + "mask/"))

    print(("=" * 10) + " Test Evaluation " + ("=" * 10))
    print(model.evaluate_segmentation(inp_images_dir=TEST_PATH + "img/",
                                      annotations_dir=TEST_PATH + "mask/"))

    model.save(MODEL_PATH)

    plt.figure(figsize=(12, 8))
    plt.title("EVALUATION")
    plt.subplot(2, 2, 1)
    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["val_loss"], label="Val_Loss")
    plt.legend()
    plt.title("Loss Evaluation")
    plt.subplot(2, 2, 2)
    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val_Accuracy")
    plt.legend()
    plt.title("Accuracy Evaluation")
    # plt.show()
    plt.savefig(FIGURE_PATH)


if __name__ == '__main__':
    # show_food("1")
    # run_data_augmentation()
    train()
