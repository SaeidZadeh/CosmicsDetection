#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 11:04:16 2023

@author: Saeid
"""
############################## Image Segmentation for cosmics detection using DeeplabV3+

# Import the required libraries
import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import decimal
from PIL import Image
import random
from tensorflow.keras import backend as K

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import precision_score, recall_score, f1_score

import seaborn as sns
import pandas as pd
from skimage import exposure

# Set the decimal precision for accurate float values
decimal.getcontext().prec = 10

# Enhance the quality of produced images
plt.rcParams['figure.dpi'] = 900
plt.rcParams['savefig.dpi'] = 900

# Load the colormap for visualization
colormap = loadmat("human_colormap.mat")["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)

# Initialization of Constants and Directories
IMAGE_SIZE = 512
BATCH_SIZE = 6
NUM_CLASSES = 2
DATA_DIR = "Data"
NUM_TRAIN_IMAGES = 31860
NUM_VAL_IMAGES = 9100
NUM_TEST_IMAGES = 4560
NUM_REAL_IMAGES = 18

# Metric Definitions for Model Evaluation

def specificity(y_true, y_pred):
    # Implementation for specificity metric
    true_negatives = K.sum((1.0 - y_true) * K.round(1.0 - y_pred))

    true_negatives_total = K.sum(1.0 - y_true)

    return true_negatives / (true_negatives_total + K.epsilon())

def precision(y_true, y_pred):
    # Implementation for precision metric
    true_positives = K.sum((y_true*1.0)*K.round(y_pred))
    predicted_positives = K.sum(K.round(y_pred))

    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    # Implementation for recall metric
    true_positives = K.sum((y_true*1.0)*K.round(y_pred))
    actual_positives = K.sum(y_true*1.0)

    return true_positives / (actual_positives + K.epsilon())

# Data Preprocessing: Identifying image and mask files for train and validation sets
DATA_DIR = "AAAAAAA"  # Replace this with the correct data directory

# Load train and validation images and masks
train_images = sorted(glob(os.path.join(DATA_DIR, "Train/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, "TrainAnnotation/*")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(DATA_DIR, "Validation/*")))[:NUM_VAL_IMAGES]
val_masks = sorted(glob(os.path.join(DATA_DIR, "ValidationAnnotation/*")))[:NUM_VAL_IMAGES]
test_images = sorted(glob(os.path.join(DATA_DIR, "Test/*")))[:NUM_TEST_IMAGES]
test_masks = sorted(glob(os.path.join(DATA_DIR, "TestAnnotation/*")))[:NUM_TEST_IMAGES]
real_images = sorted(glob(os.path.join(DATA_DIR, "Real/*")))[:NUM_REAL_IMAGES]
real_masks = sorted(glob(os.path.join(DATA_DIR, "Real/*")))[:NUM_REAL_IMAGES]

# Convert paths to strings
train_images = [str(path) for path in train_images]
train_masks = [str(path) for path in train_masks]
val_images = [str(path) for path in val_images]
val_masks = [str(path) for path in val_masks]
test_images = [str(path) for path in test_images]
test_masks = [str(path) for path in test_masks]
real_images = [str(path) for path in real_images]
real_masks = [str(path) for path in real_masks]

# Resize images and masks and transform them into tensors with batches of size 6
def read_data(image_filename):
    # Implementation for reading image data
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask

def load_data(image_list, mask_list):
    # Implementation for loading data
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask

def adjust_brightness_and_contrast(image, brightness_factor=1.1, contrast_factor=1.1):
    # Implementation for image brightness and contrast adjustment. For better visibility of low intensity pixels in images with wide range of intensities
    # Adjust brightness
    adjusted_image = exposure.adjust_gamma(image, gamma=brightness_factor)

    # Adjust contrast
    p2, p98 = np.percentile(adjusted_image, (2, 98))
    if p2!= p98:  # Some images have small intensity, their contrast should not be changed
        adjusted_image = exposure.rescale_intensity(adjusted_image, in_range=(p2, p98))

    return adjusted_image

def convert_scale_abs(image):
    # Implementation for scaling images
    image_np = image.numpy().astype(np.uint16)[:,:,0]
    adjusted_image = adjust_brightness_and_contrast(image_np)
    adjusted_image = np.repeat(adjusted_image[:, :, np.newaxis], 3, axis=2) # increase the visibility
    if adjusted_image.min()!=0:
        adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=255.0 / adjusted_image.max(), beta=0)
    return adjusted_image.astype(np.float32)

def read_image(image_path, mask=False):
    # Implementation for reading image and mask
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1, dtype=tf.uint16)
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        image = tf.cast(image, tf.int8)
        #image = tf.py_function(convert_scale_abs, [image], tf.uint8)
    else:
        image = tf.image.decode_png(image, channels=3, dtype=tf.uint16)
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        image = tf.cast(image, tf.float32)
        image = tf.py_function(convert_scale_abs, [image], tf.float32)
        image = (tf.cast(image, tf.float32) / 127.5) - 1.0
    return image

def data_generator(image_list, mask_list):
    # Implementation for data generator
    image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    mask_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)
    image_list = tf.convert_to_tensor(image_list)
    mask_list = tf.convert_to_tensor(mask_list)
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda img, mask: (tf.ensure_shape(img, image_shape), tf.ensure_shape(mask, mask_shape)))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    return dataset

real_tensors = []

def data_generator1(image_list, mask_list):
    # Implementation for data generator for ease of use of real data
    image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)  # Set the desired image shape
    mask_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)  # Set the desired mask shape

    image_list = tf.convert_to_tensor(image_list)
    mask_list = tf.convert_to_tensor(mask_list)

    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)

    # Set the element_spec of the dataset
    dataset = dataset.map(lambda img, mask: (tf.ensure_shape(img, image_shape), tf.ensure_shape(mask, mask_shape)))
    dataset = dataset.batch(1, drop_remainder=True)
    for image,_ in dataset:
        real_tensors.append(image)

    return dataset

train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)
real_dataset = data_generator(real_images, val_masks[:NUM_REAL_IMAGES ]) 
# second component are dummy values all real images will be transformed to list of tensors

# Model Definition: DeeplabV3+
def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=7,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    # Implementation for convolution block
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)  # Apply ReLU activation here
    return x

def DilatedSpatialPyramidPooling(dspp_input):
    # Implementation for Dilated Spatial Pyramid Pooling
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[1], dims[2]))(dspp_input)
    x = convolution_block(x, num_filters=256, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[1], dims[2]), interpolation="bilinear",
    )(x)

    dilations = [1, 6, 12, 18]
    dilated_convs = [
        convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=d) 
        for d in dilations
    ]

    x = layers.Concatenate(axis=-1)([out_pool] + dilated_convs)
    output = convolution_block(x, num_filters=256, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    # Implementation for DeeplabV3+ model
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    resnet50.trainable = False
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = keras.layers.Dropout(0.2)(x)
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_a = keras.layers.Dropout(0.2)(input_a)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    for _ in range(2):  # Apply convolution_block twice
        x = convolution_block(x, num_filters=256)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation = "sigmoid")(x)
    return keras.Model(inputs=model_input, outputs=model_output)

# Define the model
model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)

# Compile the model with loss and metrics
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss=loss,
    metrics=["accuracy", specificity, precision, recall],
)

# Identify checkpoints for storing the weights of the trained model for each epoch
def scheduler(epoch, lr):
    # Learning rate scheduler for better accuracy
    if epoch < 2:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

clbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)

clbk = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Change this to monitor other values
    patience=30,  # Train 30 more epochs to see if the performance improves
    verbose=0,
    mode='auto'
)  # Callback for early stopping when the model while training does not improve the performance

checkpoint_filepath = 'checkpoint.hdf5'  # The name of the file to save current model weights

mcp_save = ModelCheckpoint(
    checkpoint_filepath,
    save_weights_only=True,
    save_best_only=False,
    monitor='val_loss',
    mode='min'
)

# Train the model for 100 epochs
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    verbose=1,
    callbacks=[mcp_save, clbacks, clbk],
    shuffle=True
)

# Save the complete model
model.save_weights('filename')

# Loading the model in case other checkpoints or training weights are obtained
# model.load_weights(filepath)

# Pattern of loss and accuracy after training the model
A = history.history["loss"]
B = history.history["val_loss"]
plt.plot(A, label="Train")
plt.plot(B, label="Validation")
plt.title("Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.show()

A = history.history["accuracy"]
B = history.history["val_accuracy"]
plt.plot(A, label="Train")
plt.plot(B, label="Validation")
plt.title("Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend()
plt.show()

# Functions for visualizing the predictions of the model over the test set
def infer(model, image_tensor):
    # Implementation for model inference
    predictions = model.predict(image_tensor)
    predictions = predictions[0,:,:,1]
    return predictions

def decode_segmentation_masks(mask, colormap, n_classes):
    # Implementation for decoding segmentation masks
    r = np.zeros_like(mask).astype(np.uint16)
    g = np.zeros_like(mask).astype(np.uint16)
    b = np.zeros_like(mask).astype(np.uint16)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def get_overlay(image, colored_mask):
    # Implementation for overlaying masks on images
    image = np.array(image)
    image = image[0,:,:,0]
    overlay = cv2.addWeighted(image, 0.35, colored_mask*image.max(), 0.65, 0)
    return overlay

def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    # Implementation for plotting samples
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if i == 0:
            image = np.array(display_list[i])[0,:,:,0]
            axes[i].imshow(image)
        else:
            axes[i].imshow(display_list[i])
    plt.show()

def plot_predictions(images_list, model, N):
    # Implementation for plotting predictions
    for image_tensor in images_list[0:N]:
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_mask[prediction_mask<(1.0/(1.0+np.exp(-1.0/65535.0/2.0)))]=0
        prediction_mask[prediction_mask>=(1.0/(1.0+np.exp(-1.0/65535.0/2.0)))]=1
        overlay = get_overlay(image_tensor, prediction_mask)
        plot_samples_matplotlib(
            [image_tensor, overlay, prediction_mask], figsize=(18, 14)
        )

# Show 6 images from the real set with mask prediction
plot_predictions(real_tensors, model, 6)

# Model Evaluation
def score_segmentation(true_mask, pred_mask):
    # Implementation for scoring segmentation
    # Evaluates precision, recall and F1 score
    true_mask = np.ravel(true_mask)
    pred_mask = np.ravel(pred_mask)
    precision = precision_score(true_mask, pred_mask)
    recall = recall_score(true_mask, pred_mask)
    f1 = f1_score(true_mask, pred_mask)
    return precision, recall, f1

def calculate_iou(pred_mask, true_mask):
    # Implementation for calculating intersection over union
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Define functions required to estimate the number of cosmics
def get_adjacent_entries(matrix, row, col):
    adjacent_entries = []
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for offset in offsets:
        new_row = row + offset[0]
        new_col = col + offset[1]

        if 0 <= new_row < num_rows and 0 <= new_col < num_cols:
            if matrix[new_row][new_col] != 0:
                adjacent_entries.append((new_row, new_col))

    return adjacent_entries

def dfs(node, graph, visited, component):
    visited[node] = True
    component.append(node)

    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs(neighbor, graph, visited, component)

def find_connected_components(graph):
    components = []
    nodes = [i for i in graph.keys()]
    visited={i: False for i in nodes}

    for node in nodes:
        if not visited[node]:
            component = []
            dfs(node, graph, visited, component)
            components.append(component)

    return components

def to_graph(mat):
    graph={}
    for i in range(len(mat[:,0])):
        W=np.array(np.where(mat[i,:]!=0))[0]
        for j in W:
            Z=get_adjacent_entries(mat, i, j)
            #to_remove = [(i,j)]
            #Z=Z.remove(to_remove)
            graph[(i,j)]=Z
    return graph

# Evaluation of the trained model on test data
def evaluate_image(test_name, mask_name, model):
    test_image = Image.open(test_name, "r")
    mask_image = Image.open(mask_name, "r")
    mask_image = np.array(mask_image)
    test_image = np.array(test_image)
    image_tensor = read_image(test_name)
    prediction_mask = infer(image_tensor=image_tensor, model=model)
    prediction_mask = prediction_mask.numpy()
    binary_predicted = prediction_mask >= 1 / 255.0 / 2.0
    matrix = np.array(binary_predicted)
    graph = to_graph(matrix)
    connected_components1 = find_connected_components(graph)
    matrix = np.array(mask_image)
    graph = to_graph(matrix)
    connected_components2 = find_connected_components(graph)
    no_cosmics.append(min(1.0, len(connected_components1) / (len(connected_components2) + K.epsilon())))
    iou = calculate_iou(binary_predicted, mask_image)
    iou_scores.append(iou)
    intersection = np.sum(binary_predicted * mask_image)
    union = np.sum(binary_predicted) + np.sum(mask_image)
    dice_coefficient = (2.0 * intersection) / (union + K.epsilon())  # Add a small epsilon to avoid division by zero
    dice_coefficients.append(dice_coefficient)

    binary_ground_truth = mask_image > 0.5
    matching_pixels = (binary_predicted == binary_ground_truth).sum()
    pixel_accuracy = matching_pixels / (mask_image.shape[0] * mask_image.shape[1])  # Divide by total pixels
    tmp_accuracy = pixel_accuracy
    total_accuracy.append(tmp_accuracy)
    precision, recall, f1 = score_segmentation(binary_ground_truth, binary_predicted)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    matching_pixels = (binary_predicted == binary_ground_truth).sum()
    pixel_accuracy = matching_pixels / (test_image.shape[0] * test_image.shape[1])  # Divide by total pixels
    tmp_accuracy += pixel_accuracy
    total_accuracy.append(tmp_accuracy)
    precision, recall, f1 = score_segmentation(binary_ground_truth, binary_predicted)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    print(f"File {test_name} is evaluated!!...")

# Functions for visualizing the number of cosmics detected in an image
def show_predictions(dataset, N, model=model):
    # Implementation for number of detected cosmics
    image = dataset[N]
    prediction_mask = infer(image_tensor=image, model=model)
    prediction_mask[prediction_mask<1.0/65535.0]=0
    prediction_mask[prediction_mask>=1.0/65535.0]=1
    matrix=np.array(prediction_mask)
    graph=to_graph(matrix)
    connected_components = find_connected_components(graph)
    no_cosmics.append(len(connected_components))
    return(no_cosmics)

# Number of cosmics in a single image N-th image of the real data
show_predictions(real_tensors, N, model=model)

to_check = random.sample(range(4560), 100)  # Only check 100 images from the test set, remove if evaluating all
total_accuracy = []
iou_scores = []
dice_coefficients = []
precision_scores = []
recall_scores = []
f1_scores = []
no_cosmics = []

for i in to_check:
    test_name, mask_name = test_images[i], test_masks[i]
    evaluate_image(test_name, mask_name, model)

# Calculate mean and standard deviation of evaluation metrics
mean_iou = np.mean(iou_scores)
std_iou = np.std(iou_scores)
mean_dice_coefficient = np.mean(dice_coefficients)
std_dice_coefficient = np.std(dice_coefficients)
mean_total_accuracy = np.mean(total_accuracy)
std_total_accuracy = np.std(total_accuracy)
mean_precision = np.mean(precision_scores)
std_precision = np.std(precision_scores)
mean_recall = np.mean(recall_scores)
std_recall = np.std(recall_scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
mean_no_cosmics = np.mean(no_cosmics)
std_no_cosmics = np.std(no_cosmics)

# Show the mean and standard deviation of all the evaluation metrics defined
print("Average IoU:", mean_iou)
print("Std IoU:", std_iou)
print("Average Dice Coefficient:", mean_dice_coefficient)
print("Std Dice Coefficient:", std_dice_coefficient)
print("Average Total Accuracy:", mean_total_accuracy)
print("Std Total Accuracy:", std_total_accuracy)
print("Average Precision:", mean_precision)
print("Std Precision:", std_precision)
print("Average Recall:", mean_recall)
print("Std Recall:", std_recall)
print("Average F1 Score:", mean_f1)
print("Std F1 Score:", std_f1)
print("Average No. Cosmics:", mean_no_cosmics)
print("Std No. Cosmics:", std_no_cosmics)

# Plotting the evaluation metrics
df = pd.DataFrame({
    "IoU Acc.": iou_scores,
    "Dice Coef.": dice_coefficients,
    "Pixel Acc.": total_accuracy,
    "Precision": precision_scores,
    "Recall": recall_scores,
    "F1 Score": f1_scores,
    "no cosmics": no_cosmics
})

# Box plot for all columns in the DataFrame
ax = sns.boxplot(df, flierprops={"marker": "x"})
ax.set(xlabel='Accuracy Measure', ylabel='Accuracy',title='All Measurements')
plt.show()

# Individual box plots for each specific column
ax = sns.boxplot(df["IoU Acc."], flierprops={"marker": "x"})
ax.set(title='IoU Acc.', ylabel='Accuracy')
plt.show()

ax = sns.boxplot(df["Dice Coef."], flierprops={"marker": "x"})
ax.set(title='Dice Coef.', ylabel='Accuracy')
plt.show()

ax = sns.boxplot(df["Pixel Acc."], flierprops={"marker": "x"})
ax.set(title='Pixel Acc.', ylabel='Accuracy')
plt.show()

ax = sns.boxplot(df["Precision"], flierprops={"marker": "x"})
ax.set(title='Precision', ylabel='Accuracy')
plt.show()

ax = sns.boxplot(df["Recall"], flierprops={"marker": "x"})
ax.set(title='Recall', ylabel='Accuracy')
plt.show()

ax = sns.boxplot(df["F1 Score"], flierprops={"marker": "x"})
ax.set(title='F1 Score', ylabel='Accuracy')
plt.show()

ax = sns.boxplot(df["no cosmics"], flierprops={"marker": "x"})
ax.set(title='No. Cosmics', ylabel='Accuracy')
plt.show()

# With this implementation, we have a well-structured and complex code for image segmentation with DeeplabV3+.
# The code is used to detect cosmics as masks in image segmentation of images obtained from X-ray data.
# It utilizes a variety of libraries, custom metrics, data processing, model definition, training, and evaluation.

