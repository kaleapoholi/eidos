# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os


import io
import itertools
from packaging import version


from tensorflow import keras
from tensorflow.keras import models

from PIL.Image import *
import urllib.request

import numpy as np
import sklearn.metrics

print('Hi')
print(tf.__version__)


def show_images(train_dataset, class_names):
    

    plt.figure(figsize=(10, 10))

    for images, labels in train_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


def show_augmented_datas(train_dataset):
    for image, _ in train_dataset.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')


def data_augmentation():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    return data_augmentation


def load_dataset(url, BATCH_SIZE, IMG_SIZE):
    path_to_zip = tf.keras.utils.get_file(
        'cats_and_dogs.zip', origin=url, extract=True)

    PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

    train_dir = os.path.join(PATH, 'train')

    validation_dir = os.path.join(PATH, 'validation')

    train_dataset = image_dataset_from_directory(train_dir,
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)

    validation_dataset = image_dataset_from_directory(validation_dir,
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE)
    class_names = train_dataset.class_names
    show_images(train_dataset, class_names)

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return (train_dataset, validation_dataset, test_dataset, class_names)


def create_model(train_dataset, IMG_SIZE):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(
        1./127.5, offset=-1)

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    # Freeze the pre-trained model weights
    base_model.trainable = True

    # Trainable classification head
    maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
    #global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = maxpool_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Layer classification head with feature detector

    inputs = tf.keras.layers.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x)
    x = maxpool_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    learning_rate = 0.0001

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy']
                  )

    model.summary()
    return model


def evaluate_model(data, model):
    loss, accuracy = model.evaluate(data)
    return (loss, accuracy)

def load_model(modelfile):
    new_model = tf.keras.models.load_model(modelfile)
    return new_model


def test_model(data, model, class_names):
    #Retrieve a batch of images from the test set
    image_batch, label_batch = data.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    
    # Apply a sigmoid since our model returns logits
    pred = tf.nn.sigmoid(predictions)
    predictions = tf.where(pred < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(class_names[predictions[i]])
        plt.axis("off")

def img_test(img_path, model, class_names):
    im=urllib.request.urlretrieve(img_path, "sample.png")

    img = tf.keras.preprocessing.image.load_img("sample.png", target_size=(160, 160))
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    plt.imshow(img_tensor[0])
    plt.show()
    print(img_tensor.shape)

    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict_on_batch(images).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(classes)
    predictions = tf.where(classes < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print("Predicted class is:",class_names[predictions[0]])