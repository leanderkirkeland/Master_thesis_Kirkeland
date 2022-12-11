import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xarray as xr
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from typing import Optional, List
import copy
@tf.autograph.experimental.do_not_convert

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
    def __init__(self, embed_dim):
        super().__init__()
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
        return encoded_positions

class Backbone(layers.Layer):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.tup1 = TubeletEmbedding(dims[1],(2,2,2))
        self.tup2 = TubeletEmbedding(dims[2],(2,2,2))
        self.conv = layers.Conv3D(dims[0],1, padding = "valid")
        self.batch1 = layers.BatchNormalization()
        self.batch2 = layers.BatchNormalization()
        self.batch3 = layers.BatchNormalization()

    def get_config(self):
        config = super().get_config()
        config.update({
            "dims": self.dims,
        })
        return config
        
    def call(self, x):
        p1 = self.conv(x)
        p1 = self.batch1(p1)
        p2 = self.tup1(p1)
        p2 = self.batch2(p2)
        p3 = self.tup2(p2)
        p3 = self.batch3(p3)
        return (p3,[p1,p2,p3])

class EncoderLayer(layers.Layer):
    def __init__(self, num_heads, dim):
        super().__init__()
        self.dim = dim
        self.dim_for = dim * 4
        self.num_heads = num_heads

        self.selfatt = layers.MultiHeadAttention(num_heads = num_heads, key_dim = dim//num_heads, dropout = 0.1)
        
        self.dense1 = layers.Dense(self.dim_for)
        self.dense2 = layers.Dense(dim, activation = "relu")

        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)
        self.dropout3 = layers.Dropout(0.1)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "dim": self.dim,
        })
        return config
    def _with_pos(self,x, pos: Optional[tf.Tensor] = None):
        return x if pos is None else x+pos
    def call(self, src, pos: Optional[tf.Tensor] = None):
        q = v = self._with_pos(src,pos)
        src2 = self.selfatt(q,v,src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.dense2(self.dropout2(self.dense1(src)))
        src = src + self.dropout2(src2)
        src = self.norm3(src)
        return src

@tf.autograph.experimental.do_not_convert
class Encoder(layers.Layer):
    def __init__(self, encoder_layer, num_heads, num_layers, dim):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.num_heads = num_heads

        self.pos = PositionalEncoder(dim)
        self.layers = get_clones(encoder_layer, num_layers)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dim": self.dim,
        })
        return config

    def call(self, src, pos: Optional[tf.Tensor] = None):
        for layer in self.layers:
            src = layer(src, pos)
        return src

class DecoderLayer(layers.Layer):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.dim_for = dim*4
        self.num_heads = num_heads

        self.selfatt = layers.MultiHeadAttention(num_heads = num_heads, key_dim = dim//num_heads,dropout = 0.1)
        self.MHA = layers.MultiHeadAttention(num_heads = num_heads, key_dim = dim//num_heads,dropout = 0.1)

        self.dense1 = layers.Dense(self.dim_for)
        self.dense2 = layers.Dense(self.dim, activation = "relu")

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)
        self.dropout3 = layers.Dropout(0.1)
        self.dropout4 = layers.Dropout(0.1)


    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "dim": self.dim,
        })
        return config

    def _with_pos(self,x, pos: Optional[tf.Tensor] = None):
        return x if pos is None else x+pos

    def call(self, tgt, memory, pos: Optional[tf.Tensor] = None, quary_pos: Optional[tf.Tensor] = None):
        q = k = self._with_pos(tgt, quary_pos)

        tgt2 = self.selfatt(query = q,key = k, value = tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.MHA(query = self._with_pos(tgt,quary_pos), key = self._with_pos(memory,pos), value = memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt = self.dense2(self.dropout3(self.dense1(tgt)))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def get_clones(layer, n):
    return [copy.deepcopy(layer) for i in range(n)]

@tf.autograph.experimental.do_not_convert
class Decoder(layers.Layer):
    def __init__(self, decoder_layer, num_heads, num_layers, dim):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.num_heads = num_heads

        self.layers = get_clones(decoder_layer, num_layers)

        self.pos = PositionalEncoder(dim)
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dim": self.dim,
        })
        return config
    def call(self, tgt, memory, pos: Optional[tf.Tensor] = None):
        quary_pos = self.pos(tgt)
        for layer in self.layers:
            tgt = layer(tgt, memory, pos, quary_pos)
        return tgt


@tf.autograph.experimental.do_not_convert
class Transformer(layers.Layer):
    def __init__(self, dim, num_heads, layers):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.layers = layers
        self.Encoder = Encoder(EncoderLayer(num_heads,dim), num_heads, layers, dim)
        self.Decoder = Decoder(DecoderLayer(dim, num_heads), num_heads, layers, dim)

        self.pos = PositionalEncoder(dim)
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "layers": self.layers
        })
        return config
    def call(self, x):
        x = layers.Reshape((-1,self.dim))(x)
        pos = self.pos(x)
        memory = self.Encoder(x, pos)
        tgt = tf.zeros_like(memory)
        hs = self.Decoder(tgt, memory, pos)

        return hs, memory

class Mask_Head(layers.Layer):
    def __init__(self, inter_dims):
        super().__init__()

        self.inter_dims = inter_dims
        self.lay1 = layers.Conv3D(inter_dims[0], 3, padding="same", activation = "relu")
        self.gn1 = tfa.layers.GroupNormalization(groups=8, axis=3)
        self.lay2 = layers.Conv3D(inter_dims[0], 3, padding="same", activation = "relu")
        self.gn2 = tfa.layers.GroupNormalization(groups=8, axis=3)
        self.lay3 = layers.Conv3D(inter_dims[1], 3, padding="same", activation = "relu")
        self.gn3 = tfa.layers.GroupNormalization(groups=8, axis=3)
        self.lay4 = layers.Conv3D(inter_dims[2], 3, padding="same", activation = "relu")
        self.gn4 = tfa.layers.GroupNormalization(groups=8, axis=3)

    def get_config(self):
        config = super().get_config()
        config.update({
            "inter_dims": self.inter_dims,
        })
        return config

    def call(self, x, bbox_mask, fpns):
        x = layers.Concatenate(axis = -1)([x, bbox_mask])
        x = self.lay1(x)
        x = self.gn1(x)

        x = layers.Concatenate(axis = -1)([x,fpns[2]])
        x = self.lay2(x)
        x = self.gn2(x)
        x = layers.UpSampling3D(2)(x)

        x = layers.Concatenate(axis = -1)([x,fpns[1]])
        x = self.lay3(x)
        x = self.gn3(x)
        x = layers.UpSampling3D(2)(x)

        x = layers.Concatenate(axis = -1)([x,fpns[0]])
        x = self.lay4(x)
        out = self.gn4(x)

        return out

def create_vivit_classifier_test(
    input_shape=(64,64,32,1),
    dims = [16,32,64],
):
    inputs = layers.Input(shape=input_shape)
    features, pos = Backbone(dims)(inputs)
    hs, memory = Transformer(dim = dims[2], num_heads = 4, layers = 6)(features)
    hs = layers.Reshape((16,16,8,dims[2]))(hs)
    memory = layers.Reshape((16,16,8,dims[2]))(memory)
    outputs = Mask_Head([64,32,100])(hs, memory, pos)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_vivit_classifier_test()

train_x = []
train_y = []
valid_x = []
valid_y = []


for i in range(1):
    data = xr.open_dataset(fr"C:/Users/Leander/Skole/H2022/Datasets/Finaltrain/Train{i}.nc")
    train_x.append(data["n"])
    train_y.append(data["blob_labels"])

for i in range(1):
    data = xr.open_dataset(fr"C:/Users/Leander/Skole/H2022/Datasets/Finaltrain/Train{i+9500}.nc")
    valid_x.append(data["n"])
    valid_y.append(data["blob_labels"])

train_x = np.moveaxis(np.array(train_x).reshape(1,64,32,64,1),3,1) 
train_x = train_x/train_x.max()
train_y = np.moveaxis(np.array(train_y).reshape(1,64,32,64,1),3,1)
valid_x = np.moveaxis(np.array(valid_x).reshape(1,64,32,64,1),3,1) 
valid_x = valid_x/valid_x.max()
valid_y = np.moveaxis(np.array(valid_y).reshape(1,64,32,64,1),3,1)



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
        dataset = dataset.shuffle(batch_size * 2)

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
tf.config.run_functions_eagerly(True)
_ = model.fit(trainloader, epochs=1, validation_data=validloader, callbacks=[tensorboard_callback])

for i in range(len(model.weights)):
    model.weights[i]._handle_name = model.weights[i].name + "_" + str(i)

model.save("Showyes.h5")

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