import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xarray as xr
import matplotlib.pyplot as plt

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
    input_shape=(64,64,32,1),
    layer_norm_eps=1e-6,
    dims = [64,128,256]
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    x0 = layers.Conv3D(dims[0],1, padding = "same",kernel_regularizer='l1', activation = "relu")(inputs)
    x0 = layers.BatchNormalization()(x0)

    x1 = TubeletEmbedding(dims[1],2)(x0)
    x1 = layers.BatchNormalization()(x1)
    patches1 = layers.Conv3D(dims[1],1, padding = "same",kernel_regularizer='l1', activation = "relu")(x1)
    patches1 = layers.BatchNormalization()(patches1)

    x2 = TubeletEmbedding(dims[2],2)(patches1)
    x2 = layers.BatchNormalization()(x2)
    patches2 = layers.Conv3D(dims[2],1, padding = "same",kernel_regularizer='l1', activation = "relu")(x2)
    patches2 = layers.BatchNormalization()(patches2)

    p2 = layers.UpSampling3D((2,2,2))(patches2)
    patches1 = layers.Concatenate(axis=-1)([patches1,p2])
    patches1 = layers.Conv3D(dims[1],1, padding = "same",kernel_regularizer='l1', activation = "relu")(patches1)

    patches1 = layers.Reshape((-1,dims[1]))(patches1)
    patches2 = layers.Reshape((-1,dims[2]))(patches2)

    # Encode patches.
    encoded_patches1 = PositionalEncoder(dims[1])(patches1)
    encoded_patches2 = PositionalEncoder(dims[2])(patches2)

    list = []

    encoded_patches = encoded_patches1
    for _ in range(4):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=4, key_dim=dims[1] // 4, dropout=0.1
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

    encoded_patches = encoded_patches2
    for _ in range(8):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=dims[2] // 8, dropout=0.1
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

    # Layer normalization
    x1 = layers.LayerNormalization(epsilon=layer_norm_eps)(list[0])
    x2 = layers.LayerNormalization(epsilon=layer_norm_eps)(list[1])
    x1 = layers.Reshape((32,32,16,dims[1]))(x1)
    x2 = layers.Reshape((16,16,8,dims[2]))(x2)

    x1 = layers.Concatenate(axis=-1)([layers.Reshape((32,32,16,dims[1]))(patches1),x1])#try
    x1 = layers.Conv3D(dims[1],3,padding = "same",kernel_regularizer='l1', activation = "relu")(x1)
    x1 = layers.BatchNormalization()(x1)

    x2 = layers.Concatenate(axis=-1)([layers.Reshape((16,16,8,dims[2]))(patches2),x2])#try
    x2 = layers.Conv3D(dims[2],3,padding = "same",kernel_regularizer='l1', activation = "relu")(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.UpSampling3D((2,2,2))(x2)
    x2 = layers.Conv3D(dims[1],3,padding = "same",kernel_regularizer='l1', activation = "relu")(x2)
    x2 = layers.BatchNormalization()(x2)

    x1 = layers.UpSampling3D((2,2,2))(x1)
    x1 = layers.Conv3D(dims[0],3,padding = "same",kernel_regularizer='l1', activation = "relu")(x1)
    x2 = layers.UpSampling3D((2,2,2))(x2)
    x2 = layers.Conv3D(dims[0],3,padding = "same",kernel_regularizer='l1', activation = "relu")(x2)

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

train_x = xr.open_dataset("train_x").to_array().to_numpy()[0]
train_y = xr.open_dataset("train_y").to_array().to_numpy()[0]
valid_x = xr.open_dataset("valid_x").to_array().to_numpy()[0]
valid_y = xr.open_dataset("valid_y").to_array().to_numpy()[0]

valid_y.max()

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
    batch_size: int = 1,
):
    """Utility function to prepare the dataloader."""
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    if loader_type == "train":
        dataset = dataset.shuffle(1 * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataloader

trainloader = prepare_dataloader(train_x, train_y, "train")
validloader = prepare_dataloader(valid_x, valid_y, "valid")

optimizer = keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    )


log_dir = "logs/fit/64x64x32" 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

_ = model.fit(trainloader, epochs=200, validation_data=validloader, callbacks=[tensorboard_callback])

model.save("Transformer_64x64x32")

model.evaluate(testloader)

y = np.argmax(model.predict(testloader),-1)

plt.imsave(test_x[0,32,:,:,0])
plt.show()
plt.imshow(y[0,32])
plt.show()
plt.imshow(test_y[0,32,:,:,0])
plt.show()

def animate(i: int) -> None:
    arr = frames[i]
    vmax = np.max(arr)
    vmin = np.min(arr)
    im.set_data(arr)
    im.set_clim(vmin, vmax)
    tx.set_text(f"t = {i*dt:.2f}")

def gif(dataset, name = "Gif"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")

    frames = []

    for k in range(dataset.shape[0]):
        frame = dataset[k,:,:]
        frames.append(frame)

    cv0 = frames[0]
    im = ax.imshow(cv0, origin="lower")
    fig.colorbar(im, cax=cax)
    tx = ax.set_title("t = 0")

    dt= 0.1
    def animate(i: int) -> None:
        arr = frames[i]
        vmax = np.max(arr)
        vmin = np.min(arr)
        im.set_data(arr)
        im.set_clim(vmin, vmax)
        tx.set_text(f"t = {i*dt:.2f}")

    ani = FuncAnimation(
        fig, animate, frames=dataset.shape[0], interval=100)
    ani.save(f"{name}.gif", fps=10)

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

gif(y[0],name="16x16x16_advanced_120epochs")
gif(test_y[0,:,:,:,0],name="16x16x16_advanced__120epochs_label")

acc_y = y.reshape((-1,16,16))
acc_lab = test_y.reshape((-1,16,16))

iou1 = IoU2(acc_lab, acc_y, 5)
iou2 = tf.keras.metrics.IoU(101,[np.arange(np.arange(101))])(y,test_y[:,:,:,:,0])
acc1 = accuracy(acc_lab,acc_y,5)
acc2 = model.evaluate(testloader)

metrics = [iou1,iou2,acc1,acc2]

with open('metrics.txt', 'w') as fp:
    for item in metrics:
        # write each item on a new line
        fp.write("%s\n" % item)