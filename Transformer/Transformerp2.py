from cgi import test
import os
import io
import imageio
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xarray as xr
import matplotlib.pyplot as plt

# DATA
BATCH_SIZE = 8
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (16, 16, 16, 1)
NUM_CLASSES = 5

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 60

# TUBELET EMBEDDING
PATCH_SIZE = (2, 2, 2)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 2
NUM_LAYERS = 8

train_x = []
train_y = []
valid_x = []
valid_y = []
test_x = []
test_y = []

for i in range(2900):
    data = xr.open_dataset(f"Datasets/16x16x16_100/16x16x16_100{i}.nc")
    train_x.append(data["n"])
    train_y.append(data["blob_labels"])

for i in range(50):
    data = xr.open_dataset(f"Datasets/16x16x16_100/16x16x16_100{i+2900}.nc")
    valid_x.append(data["n"])
    valid_y.append(data["blob_labels"])

for i in range(50):
    data = xr.open_dataset(f"Datasets/16x16x16_100/16x16x16_100{i+2950}.nc")
    test_x.append(data["n"])
    test_y.append(data["blob_labels"])

np.array(test_y).shape

train_x = np.moveaxis(np.array(train_x).reshape(2900,16,16,16,1),3,1) 
train_x = train_x/train_x.max()
train_y = np.moveaxis(np.array(train_y).reshape(2900,16,16,16,1),3,1)
valid_x = np.moveaxis(np.array(valid_x).reshape(50,16,16,16,1),3,1) 
valid_x = valid_x/valid_x.max()
valid_y = np.moveaxis(np.array(valid_y).reshape(50,16,16,16,1),3,1)
test_x = np.moveaxis(np.array(test_x).reshape(50,16,16,16,1),3,1) 
test_x = test_x/test_x.max()
test_y = np.moveaxis(np.array(test_y).reshape(50,16,16,16,1),3,1)

@tf.function
def preprocess(frames: tf.Tensor, label: tf.Tensor):
    """Preprocess the frames tensors and parse the labels."""
    # Preprocess images
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],  # The new axis is to help for further processing with Conv3D layers
        tf.float32,
    )
    # Parse label
    label = tf.cast(label, tf.float32)
    return frames, label


def prepare_dataloader(
    videos: np.ndarray,
    labels: np.ndarray,
    loader_type: str = "train",
    batch_size: int = BATCH_SIZE,
):
    """Utility function to prepare the dataloader."""
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    if loader_type == "train":
        dataset = dataset.shuffle(BATCH_SIZE * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataloader

trainloader = prepare_dataloader(train_x, train_y, "train")
validloader = prepare_dataloader(valid_x, valid_y, "valid")
testloader = prepare_dataloader(test_x, test_y, "test")


class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=3,
            strides=patch_size,
            padding="same",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        return projected_patches

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens


def create_vivit_classifier_test(
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
    dims = [64,128,256]
):
    # Get the input layer
    print("backbone")
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    x0 = layers.Conv3D(dims[0],1, padding = "same",kernel_regularizer='l1', activation = "relu")(inputs)
    x0 = layers.BatchNormalization()(x0)
    print(1)
    x1 = TubeletEmbedding(dims[1],2)(x0)
    x1 = layers.BatchNormalization()(x1)
    patches1 = layers.Conv3D(dims[1],1, padding = "same",kernel_regularizer='l1', activation = "relu")(x1)
    patches1 = layers.BatchNormalization()(patches1)
    print(2)
    x2 = TubeletEmbedding(dims[2],2)(patches1)
    x2 = layers.BatchNormalization()(x2)
    patches2 = layers.Conv3D(dims[2],1, padding = "same",kernel_regularizer='l1', activation = "relu")(x2)
    patches2 = layers.BatchNormalization()(patches2)
    print(3)
    p2 = layers.UpSampling3D((2,2,2))(patches2)
    patches1 = layers.Concatenate(axis=-1)([patches1,p2])
    patches1 = layers.Conv3D(dims[1],1, padding = "same",kernel_regularizer='l1', activation = "relu")(patches1)
    print(4)

    patches1 = layers.Reshape((-1,dims[1]))(patches1)
    patches2 = layers.Reshape((-1,dims[2]))(patches2)
    print(5)

    # Encode patches.
    encoded_patches1 = PositionalEncoder(dims[1])(patches1)
    encoded_patches2 = PositionalEncoder(dims[2])(patches2)
    print(encoded_patches1.shape, encoded_patches2.shape)

    list = []
    
    print("transformer")

    encoded_patches = encoded_patches1
    for _ in range(4):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dims[1] // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=dims[1] * 4, activation=tf.nn.gelu),
                layers.Dense(units=dims[1], activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])
    list.append(encoded_patches)

    print(encoded_patches.shape)



    encoded_patches = encoded_patches2
    for _ in range(8):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dims[2] // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=dims[2] * 4, activation=tf.nn.gelu,kernel_regularizer='l1'),
                layers.Dense(units=dims[2], activation=tf.nn.gelu,kernel_regularizer='l1'),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])
    list.append(encoded_patches)
    print(encoded_patches.shape)

    print("segment")
    # Layer normalization
    x1 = layers.LayerNormalization(epsilon=layer_norm_eps)(list[0])
    x2 = layers.LayerNormalization(epsilon=layer_norm_eps)(list[1])
    x1 = layers.Reshape((8,8,8,dims[1]))(x1)
    x2 = layers.Reshape((4,4,4,dims[2]))(x2)
    print(1)
    x1 = layers.Concatenate(axis=-1)([layers.Reshape((8,8,8,dims[1]))(patches1),x1])#try
    x1 = layers.Conv3D(dims[1],3,padding = "same",kernel_regularizer='l1', activation = "relu")(x1)
    x1 = layers.BatchNormalization()(x1)
    print(2)
    x2 = layers.Concatenate(axis=-1)([layers.Reshape((4,4,4,dims[2]))(patches2),x2])#try
    x2 = layers.Conv3D(dims[2],3,padding = "same",kernel_regularizer='l1', activation = "relu")(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.UpSampling3D((2,2,2))(x2)
    x2 = layers.Conv3D(dims[1],3,padding = "same",kernel_regularizer='l1', activation = "relu")(x2)
    x2 = layers.BatchNormalization()(x2)
    print(3)
    x1 = layers.UpSampling3D((2,2,2))(x1)
    x1 = layers.Conv3D(dims[0],3,padding = "same",kernel_regularizer='l1', activation = "relu")(x1)
    x2 = layers.UpSampling3D((2,2,2))(x2)
    x2 = layers.Conv3D(dims[0],3,padding = "same",kernel_regularizer='l1', activation = "relu")(x2)
    print(4)
    #representation = layers.Add()([x1,x2])
    representation = layers.Concatenate(axis=-1)([x0,x1,x2])

    representation = layers.Conv3D(200,1, padding = "same",kernel_regularizer='l1', activation = "relu")(representation)
    representation = layers.BatchNormalization()(representation)
    outputs = layers.Conv2D(100,1,1)(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_vivit_classifier_test()

model.summary()

np.random.normal(1,0, size= 100)

def create_vivit_classifier(
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    x = layers.Conv3D(32,1, padding = "same",kernel_regularizer='l1', activation = "relu")(inputs)
    x = layers.BatchNormalization()(x)

    x1 = TubeletEmbedding(32,2)(x)
    x1 = layers.BatchNormalization()(x1)
    patches1 = layers.Conv3D(32,1, padding = "same",kernel_regularizer='l1', activation = "relu")(x1)
    patches1 = layers.BatchNormalization()(patches1)

    x2 = TubeletEmbedding(32,2)(patches1)
    x2 = layers.BatchNormalization()(x2)
    patches2 = layers.Conv3D(32,1, padding = "same",kernel_regularizer='l1', activation = "relu")(x2)
    patches2 = layers.BatchNormalization()(patches2)

    p2 = layers.UpSampling3D((2,2,2))(patches2)
    patches1 = layers.Add()([patches1,p2])

    p2 = layers.UpSampling3D((2,2,2))(patches1)
    patches0 = layers.Add()([x,p2])

    patches01 = layers.Reshape((-1,32))(patches0)
    patches11 = layers.Reshape((-1,32))(patches1)
    patches21 = layers.Reshape((-1,32))(patches2)


    # Encode patches.
    encoded_patches0 = PositionalEncoder(32)(patches01)
    encoded_patches1 = PositionalEncoder(32)(patches11)
    encoded_patches2 = PositionalEncoder(32)(patches21)

    list = []

    # Create multiple layers of the Transformer block.
    encoded_patches = encoded_patches0
    for _ in range(1):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=32 // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=32 * 4, activation=tf.nn.gelu),
                layers.Dense(units=32, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])
    list.append(encoded_patches)
    


    encoded_patches = encoded_patches1
    for _ in range(4):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=32 // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=32 * 4, activation=tf.nn.gelu),
                layers.Dense(units=32, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])
    list.append(encoded_patches)



    encoded_patches = encoded_patches2
    for _ in range(8):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=32 // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=32 * 4, activation=tf.nn.gelu,kernel_regularizer='l1'),
                layers.Dense(units=32, activation=tf.nn.gelu,kernel_regularizer='l1'),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])
    list.append(encoded_patches)



    # Layer normalization and Global average pooling.
    x0 = layers.LayerNormalization(epsilon=layer_norm_eps)(list[0])
    x1 = layers.LayerNormalization(epsilon=layer_norm_eps)(list[1])
    x2 = layers.LayerNormalization(epsilon=layer_norm_eps)(list[2])
    x0 = layers.Reshape((16,16,16,32))(x0)
    x1 = layers.Reshape((8,8,8,32))(x1)
    x2 = layers.Reshape((4,4,4,32))(x2)
    patches01 = layers.Reshape((16,16,16,32))(patches01)

    x0 = layers.Concatenate(axis=-1)([patches01,x0])
    x0 = layers.Conv3D(32,3,padding = "same",kernel_regularizer='l1', activation = "relu")(x0)
    x0 = layers.BatchNormalization()(x0)

    x1 = layers.Concatenate(axis=-1)([patches1,x1])#try
    x1 = layers.Conv3D(32,3,padding = "same",kernel_regularizer='l1', activation = "relu")(x1)
    x1 = layers.BatchNormalization()(x1)

    x2 = layers.Concatenate(axis=-1)([patches2,x2])#try
    x2 = layers.Conv3D(32,3,padding = "same",kernel_regularizer='l1', activation = "relu")(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.UpSampling3D((2,2,2))(x2)
    x2 = layers.Conv3D(32,3,padding = "same",kernel_regularizer='l1', activation = "relu")(x2)
    x2 = layers.BatchNormalization()(x2)

    x1 = layers.UpSampling3D((2,2,2))(x1)
    x1 = layers.Conv3D(32,3,padding = "same",kernel_regularizer='l1', activation = "relu")(x1)
    x2 = layers.UpSampling3D((2,2,2))(x2)
    x2 = layers.Conv3D(32,3,padding = "same",kernel_regularizer='l1', activation = "relu")(x2)

    #representation = layers.Add()([x1,x2])
    representation = layers.Concatenate(axis=-1)([x0,x1,x2])

    representation = layers.Conv3D(32,1, padding = "same",kernel_regularizer='l1', activation = "relu")(representation)
    representation = layers.BatchNormalization()(representation)
    outputs = layers.Conv2D(100,1,1)(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_vivit_classifier()

model.summary()

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    )

import os
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

_ = model.fit(trainloader, epochs=10, validation_data=validloader, callbacks=[tensorboard_callback])

model.save("Transformer_16x16_advanced_120epochs")

model = tf.keras.models.load_model("Transformer_16x16_advanced_120epochs")

model.evaluate(testloader)

y = np.argmax(model.predict(testloader),-1)

test_y[1].max()

plt.imshow(test_x[0,8,:,:,0])
plt.show()
plt.imshow(y[0,8])
plt.show()
plt.imshow(test_y[0,8,:,:,0])
plt.show()

import Code.LSTM.utils

Code.LSTM.utils.gif(y[0],name="16x16x16_advanced_120epochs")
Code.LSTM.utils.gif(test_y[0,:,:,:,0],name="16x16x16_advanced__120epochs_label")

def IoU2(labels, predictions, num_c):
  # mean Intersection over Union
  # Mean IoU = TP/(FN + TP + FP)
  IoU_list = []
  for c in range(num_c):
    TP = np.sum( (labels==c)&(predictions==c) )
    FP = np.sum( (labels!=c)&(predictions==c) )
    FN = np.sum( (labels==c)&(predictions!=c) )

    IoU = TP/(TP + FP + FN)
    IoU_list.append(IoU)

  return IoU_list

def accuracy(labels,predictions,num_c):
    acc_list = []           
    for c in range(num_c):
        TP = np.sum( (labels==c)&(predictions==c))
        FP = np.sum( (labels!=c)&(predictions==c) )
        FN = np.sum( (labels==c)&(predictions!=c) )
        TN = np.sum( (labels!=c)&(predictions!=c))
        acc = (TP)/(TP+FN)
        acc_list.append(acc)
    return acc_list

acc_y = y.reshape((-1,16,16))
acc_lab = test_y.reshape((-1,16,16))

iou1 = IoU2(acc_lab, acc_y, 5)
iou2 = tf.keras.metrics.IoU(100,[0,1,2,3,4])(y,test_y[:,:,:,:,0])
acc1 = accuracy(acc_lab,acc_y,5)
acc2 = model.evaluate(testloader)

metrics = [iou1,iou2,acc1,acc2]

with open('metrics.txt', 'w') as fp:
    for item in metrics:
        # write each item on a new line
        fp.write("%s\n" % item)