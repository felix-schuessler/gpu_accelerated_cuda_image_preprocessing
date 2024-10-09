import os
import time
import logging
import argparse
import random
import warnings

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from pycuda.compiler import SourceModule

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate

from util import *

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

rotate_kernel_code = load_kernel_code(file_path='./src/rotate_kernel.cu')
mod = SourceModule(rotate_kernel_code)
rotate_kernel = mod.get_function("rotate_kernel")

@time_exec
def rotate_images_with_cuda(input_images, angle):
    """Rotates a batch of images using a CUDA kernel.

    This function efficiently rotates a batch of images specified in `input_images`
    by a given `angle` in degrees using a CUDA kernel. The rotation operation
    is performed on the GPU for faster processing.

    Args:
        input_images (list[np.ndarray]): A list of NumPy arrays representing the input images.
            Each image is expected to be a 3-channel (RGB) array with shape (height, width, 3).
            All images in the batch are assumed to have the same size.
        angle (float): The rotation angle in degrees.

    Returns:
        np.ndarray: A NumPy array containing the rotated images. The output array has the
            same shape as the list of input images (number of images, height, width, 3).
    """
    batch_size = len(input_images)

    height, width, _ = input_images[0].shape

    input_image_batch = np.concatenate([img.flatten() for img in input_images]).astype(np.uint8)
    output_image_batch = np.empty_like(input_image_batch)

    block_size = (16, 16, 1)
    grid_size = ((width + block_size[0] - 1) // block_size[0], 
                 (height + block_size[1] - 1) // block_size[1], 
                 batch_size)

    rotate_kernel(
        drv.In(input_image_batch), drv.Out(output_image_batch), 
        np.int32(width), np.int32(height), np.int32(batch_size), np.float32(np.radians(angle)),
        block=block_size, grid=grid_size
    )

    output_images = [output_image_batch[i * height * width * 3:(i + 1) * height * width * 3].reshape((height, width, 3)) for i in range(batch_size)]
    
    return np.array(output_images)

@time_exec
def rotate_images_with_pillow(image_array, angle):
    img_dim = image_array.shape[1]
    return np.array([
        np.array(Image.fromarray(img).rotate(angle, expand=True).resize((img_dim, img_dim), Image.LANCZOS))
        for img in image_array
    ])

@time_exec
def rotate_images_with_scipy(images, angle):
    rotated_images = np.empty_like(images)
    for i in range(len(images)):
        rotated_images[i] = rotate(images[i], angle, reshape=False, axes=(1, 0))
    return rotated_images


def save_augmented_image(images):

    augmented_dir = f'{OUTPUT_DIR}/augmented/{ARGS.augment}'
    os.makedirs(augmented_dir, exist_ok=True)
    print(f"Saving Augmented image in: {augmented_dir}")

    image = images[0] 
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(image)
    image_path = f"{augmented_dir}/img.png"
    print(f"Saving image to: {image_path}")
    pil_img.save(image_path)


def plot_training_history(history, augmentation_method):

    augmented_dir = f'{AUGMENTED_DIR}/{augmentation_method}'
    os.makedirs(augmented_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(augmented_dir, 'model_accuracy.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(augmented_dir, 'model_loss.png'))
    plt.close()


def dataset_generator(path, img_dim, num_img=None):
    """
    Generator to yield images and their labels repeatedly until a specific number of samples is reached.
    
    :param path: Directory path containing images.
    :param img_dim: dimension of the image, will be passed to tuple (img_dim, img_dim).
    :param num_img: Total number of images to yield (if specified), 
        will repeat if more images are specified than the directory contains.
        will stop early if less images are specified than contained in the directory.
    """
    current_count = 0
    
    image_files = [f for f in os.listdir(path) if f.lower().endswith('.jpg')]

    if num_img is None:
        num_img = len(image_files)
    
    labels = [filename.split('_')[0] for filename in image_files]
    labels_df = pd.DataFrame(labels, columns=['label'])
    
    one_hot_labels = pd.get_dummies(labels_df['label']).values

    while True:
        for i, filename in enumerate(image_files):
            img = Image.open(os.path.join(path, filename))
            img = img.resize((img_dim, img_dim))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                img = img.convert('RGB')

            yield np.array(img), one_hot_labels[i]
            
            current_count += 1
            
            if num_img is not None and current_count >= num_img:
                return


def prepare_dataset(x=None, y=None, generator=None, prefetch=True, repeat=False):
    if generator is not None:
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(ARGS.img_dim, ARGS.img_dim, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32)
            )
        )
    elif x is not None and y is not None:
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    dataset = dataset.cache().shuffle(buffer_size=1000).batch(ARGS.batch_size)
    if repeat:
        dataset = dataset.repeat()
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def parse_arguments():
    """
    Parse command-line arguments and return the configuration.
    Additionally, print out the complete configuration, including defaults.
    """
    
    parser = argparse.ArgumentParser(description='Image Augmentation using CUDA or Pillow.')

    parser.add_argument(
        '--augment',
        type=str,
        choices=list(AUGMENT_CHOICES.keys()),
        default=None,
        help="Use 'cuda' for CUDA-based rotation, 'pillow' for Pillow-based rotation, or leave empty for no augmentation."
    )
    parser.add_argument('--img_dim', type=int, default=200, help="Image dimensions (height and width). Default is 200.")
    parser.add_argument('--num_img', type=int, default=240, help="Number of images to process. Default is 240.")
    parser.add_argument('--batch_size', type=int, default=50, help="Batch size for training. Default is 50.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs for training. Default is 50.")
    parser.add_argument('--benchmark', action='store_true', help="Flag to enable benchmarking.")
    
    ARGS = parser.parse_args()

    ARGS.num_val_img = 60

    def print_config(param_name, param_value, default_value):
        if param_value == default_value:
            print(f"{param_name}: {param_value} (using default)")
        else:
            print(f"{param_name}: {param_value}")

    print("\nConfiguration:")
    print_config("Augmentation Method", ARGS.augment, None)
    print_config("Image Dimension", ARGS.img_dim, 200)
    print_config("Number of Training Images", ARGS.num_img, 240)
    print_config("Number of Valiation Images", ARGS.num_val_img, 60)
    print_config("Batch Size", ARGS.batch_size, 50)
    print_config("Epochs", ARGS.epochs, 50)
    print(f"Benchmark Run: {ARGS.benchmark}\n")

    return ARGS


if __name__ == "__main__":

    OUTPUT_MAPPING = {0: 'apple', 1: 'banana', 2: 'mixed', 3: 'orange'}
    NUM_CLASSES = len(OUTPUT_MAPPING)

    AUGMENT_CHOICES = {
        'cuda': rotate_images_with_cuda,
        'pillow': rotate_images_with_pillow,
        'scipy': rotate_images_with_scipy
    }

    ARGS = parse_arguments()


    batch_images, batch_labels = [], []
    augmentation_time = 0

    all_images = np.empty((0, ARGS.img_dim, ARGS.img_dim, 3), dtype=np.uint8)
    all_labels = np.empty((0, NUM_CLASSES), dtype=np.float32)

    train_dataset = None
    
    if ARGS.augment not in list(AUGMENT_CHOICES.keys()):

        train_dataset = prepare_dataset(
            generator=lambda: dataset_generator(TRAIN_DIR, img_dim=ARGS.img_dim, num_img=ARGS.num_img),
            repeat=True
        )

    else:

        for img, label in dataset_generator(TRAIN_DIR, img_dim=ARGS.img_dim, num_img=ARGS.num_img):
            batch_images.append(img)
            batch_labels.append(label)

            if len(batch_images) == ARGS.batch_size:
                        
                print(f'Start Augmentation ({ARGS.augment})')

                images_batch = np.array(batch_images)
                labels_batch = np.array(batch_labels)
                
                rotation_range=360
                random_angle = np.random.randint(rotation_range * -1, rotation_range)

                if ARGS.augment == 'cuda':
                    augmented_images, aug_time = rotate_images_with_cuda(images_batch, random_angle)
                if ARGS.augment == 'pillow':
                    augmented_images, aug_time = rotate_images_with_pillow(images_batch, random_angle)
                if ARGS.augment == 'scipy':
                    augmented_images, aug_time = rotate_images_with_scipy(images_batch, random_angle)
                
                if not ARGS.benchmark:
                    save_augmented_image(augmented_images)

                augmentation_time += aug_time
                
                images = np.concatenate((images_batch, augmented_images), axis=0)
                labels = np.tile(labels_batch, (2, 1))

                all_images = np.concatenate((all_images, images), axis=0)
                all_labels = np.concatenate((all_labels, labels), axis=0)

                batch_images, batch_labels = [], []
        
        print(f'Augmentation Time ({ARGS.augment}): {augmentation_time}\n')

        if ARGS.benchmark:
            os.makedirs(BENCHMARK_DIR, exist_ok=True)

            results = []
            results.append({
                IMAGE_DIMENSION: ARGS.img_dim,
                NUMBER_OF_IMAGES: ARGS.num_img,
                BATCH_SIZE: ARGS.batch_size,
                AUGMENTATION_METHOD: ARGS.augment,
                TIME_SEC: augmentation_time,
            })
            results_df = pd.DataFrame(results)
            
            if os.path.exists(BENCHMARK_RESULTS_CSV):
                df = pd.read_csv(BENCHMARK_RESULTS_CSV)
                results_df = pd.concat([df, results_df], ignore_index=True)

            results_df.to_csv(BENCHMARK_RESULTS_CSV, index=False)
            exit(0)
        
        train_dataset = prepare_dataset(x=all_images, y=all_labels, repeat=True)

    print("Trains-Dataset Specs:", train_dataset.element_spec)

    model = Sequential([
        Conv2D(kernel_size=(3,3), filters=32, activation='tanh', input_shape=(ARGS.img_dim, ARGS.img_dim,3,)),
        Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'),
        MaxPool2D(2, 2),
        Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'),
        MaxPool2D(2, 2),
        Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'),
        Flatten(),
        Dense(20, activation='relu'),
        Dense(15, activation='relu'),
        Dense(4,activation = 'softmax')
    ])
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
    model.summary()


    validation_dataset = prepare_dataset(
        generator=lambda: dataset_generator(path=TEST_DIR, img_dim=ARGS.img_dim, num_img=ARGS.num_val_img),
        repeat=False
    )

    print("Train_dataset Specs:", train_dataset.element_spec)
    print("Validation_dataset Specs:", validation_dataset.element_spec)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=ARGS.epochs,
        steps_per_epoch = ARGS.num_img // ARGS.batch_size,
        validation_steps=ARGS.num_val_img // ARGS.batch_size,
        verbose=1
    )

    aug_meth = ARGS.augment if ARGS.augment != '' else 'vanilla'
    plot_training_history(history=history, augmentation_method=aug_meth)
    
    validation_generator = lambda: dataset_generator(path=TEST_DIR, img_dim=ARGS.img_dim, num_img=None)
    validation_dataset = prepare_dataset(x=all_images, y=all_labels, generator=validation_generator, repeat=False)

    for checkImages, checklabels in validation_dataset.take(1):
        checkImage = checkImages[0].numpy()
        checklabel = checklabels[0].numpy()

        predict = model.predict(np.expand_dims(checkImage, axis=0))

        print("Actual :- ", checklabel)
        print("Predicted :- ", OUTPUT_MAPPING[np.argmax(predict)])

    evaluate = model.evaluate(validation_dataset)
    print(f'Evaluate: {evaluate}')
