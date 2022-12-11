from cProfile import label
from turtle import color
import tensorflow as tf
import numpy as np 
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from Code.LSTM.Model import LSTM32x64
import xarray as xr
import matplotlib.pyplot as plt

"""
import pickle

infile = open("CMod_data.pickle",'rb')
new_dict = pickle.load(infile)
infile.close()

new_dict.keys()

brt_arr = np.flip(np.moveaxis(new_dict["brt_arr"],1,0)[:,:32,:],1)
brt_arr = brt_arr/brt_arr.max()
r_arr = new_dict["r_arr"]
z_arr = new_dict["z_arr"]
R_LCFS = new_dict["R_LCFS"]
Z_LCFS = new_dict["Z_LCFS"]
R_LIM = new_dict["R_LIM"]
Z_LIM = new_dict["Z_LIM"]

plt.imshow(brt_arr[:,:,100])
plt.show()

plt.imshow(new_dict["brt_arr"][:,:,100])
plt.show()

brt_arr.shape[0]


plt.contourf(r_arr,z_arr,new_dict["brt_arr"][:,:,100])
plt.plot(R_LCFS,Z_LCFS,color="r")
plt.plot(R_LIM,Z_LIM,"r")
plt.show()

val = np.zeros(brt_arr.shape)

for i in range(brt_arr.shape[0]):
    for j in range(brt_arr.shape[1]):
        m = brt_arr[i,j].mean()
        for k in range(brt_arr.shape[2]):
            val[i,j,k] = brt_arr[i,j,k] - m

val = (val-np.min(val))/(np.max(val)-np.min(val))

np.min(val)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable


fig = plt.figure()
ax = fig.add_subplot(111)
div = make_axes_locatable(ax)
cax = div.append_axes("right", "5%", "5%")

frames = []

for k in range(750):
    frame = val[:,:,k]
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
    fig, animate, frames=750, interval=100)
ani.save("normalized.gif", writer="ffmpeg", fps=10)

train_x = []
train_y = []
test_x = []
test_y = []

for i in range(1):
    data = xr.open_dataset(f"Datasets/32x64test{i}.nc")
    train_x.append(data["n"])
    train_y.append(data["blob_labels"])

np.array(train_x).shape

for i in range(20):
    data = xr.open_dataset(f"Datasets/x64test{i+180}.nc").to_array()
    test_x.append(data[0])
    test_y.append(data[1])

train_x = np.moveaxis(np.array(train_x).reshape(180,64,64,200,1),3,1)
train_y = np.moveaxis(np.array(train_y).reshape(180,64,64,200,1),3,1)
test_x = np.moveaxis(np.array(test_x).reshape(20,64,64,200,1),3,1)
test_y = np.moveaxis(np.array(test_y).reshape(20,64,64,200,1),3,1)

model = LSTM64x64()

weights = np.where(train_y == 0, 0.05, 0.5)

model.fit(train_x, train_y, epochs=60, batch_size=4, sample_weight=weights)

model.save("LSTMtest")

model = load_model("LSTMtest")

x = model.predict(test_x[0:2])

x = np.argmax(x,-1)

plt.imshow(x[0,50,:,:])
plt.show()

plt.imshow(test_y[0,50,:,:])
plt.show()

model.summary()
"""

from cProfile import label
from turtle import color
import tensorflow as tf
import numpy as np 
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from Code.LSTM.Model import LSTM32x64
from Code.LSTM.utils import IoU2, accuracy, gif, normalize
import xarray as xr
import matplotlib.pyplot as plt
import os
import datetime

with open("EXP2\Experimental_50.npy","rb") as f:
    expx1 = np.load(f)

model = load_model("LSTMtest2")

expx1 = expx1.reshape(2048,64,32,1)[:2000].reshape(20,100,64,32,1)
x = model.predict(expx1)

with open("Exp_LSTM.npy", "wb") as f:
    np.save(f, x)

train_x = []
train_y = []
val_x = []
val_y = []
test_x = []
test_y = []

for i in range(3000):
    data = xr.open_dataset(f"Datasets3/32x64test{i}.nc")
    train_x.append(data["n"])
    train_y.append(data["blob_labels"])

for i in range(500):
    data = xr.open_dataset(f"Datasets3/32x64test{i+3000}.nc").to_array()
    val_x.append(data[0])
    val_y.append(data[1])


train_x = np.moveaxis(np.array(train_x).reshape(3000,64,32,100,1),3,1) #makes it shape (n, 100, 64, 32, 1)
train_y = np.moveaxis(np.array(train_y).reshape(3000,64,32,100,1),3,1)
val_x = np.moveaxis(np.array(val_x).reshape(500,64,32,100,1),3,1)
val_y = np.moveaxis(np.array(val_y).reshape(500,64,32,100,1),3,1)

train_x = train_x/np.max(train_x)
val_x = val_x/np.max(val_x)

I = []
A = []
L = []

"""model = load_model("LSTMtest1")"""

model = LSTM32x64()

model.summary()

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='LSTM.png', show_shapes=True, show_layer_names=True)

import visualkeras
visualkeras.layered_view(model)

visualkeras.layered_view(model, legend=True) # without custom font
from PIL import ImageFont
import PIL
font = ImageFont.truetype("arial.ttf", 12)
x = visualkeras.layered_view(model, legend=True, font=font) # selected font
plt.imshow(x)
plt.xticks([])
plt.yticks([])
plt.show()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

weights = np.where(train_y == 0, 0.1, 0.9)
model.fit(train_x, train_y, epochs=200, batch_size=8, sample_weight=weights, validation_data=(val_x, val_y), callbacks=[tensorboard_callback])

model.save("LSTMtest4")

test_x = []
test_y = []

for i in range(500):
    data = xr.open_dataset(f"Datasets3/32x64test{i+3500}.nc").to_array()
    test_x.append(data[0])
    test_y.append(data[1])


test_x = np.moveaxis(np.array(test_x).reshape(500,64,32,100,1),3,1)
test_y = np.moveaxis(np.array(test_y).reshape(500,64,32,100,1),3,1)

test_x = test_x/np.max(test_x)

from Code.LSTM.utils import normalize, rightsize
import pickle

infile = open("CMod_data.pickle",'rb')
new_dict = pickle.load(infile)
infile.close()

new_dict.keys()

z_arr = new_dict["z_arr"]

brt_arr = new_dict["brt_arr"]
r_arr = new_dict["r_arr"]

brt_arr.shape
r_arr.shape
z_arr

new_dict["R_LCFS"].shape

brt = rightsize(brt_arr)

z = normalize(brt_arr, normal="gauss",cutoff=20)

z.mean()

plt.imshow(z[:,:,50])
plt.show()



plt.contourf(r_arr,z_arr, brt_arr[:,:,50])
plt.contour(r_arr,z_arr,z[:,:,50])
plt.plot(new_dict["R_LCFS"], new_dict["Z_LCFS"])
plt.plot(new_dict["R_LIM"], new_dict["Z_LIM"])
plt.show()

z = z/np.max(z)
for i in range(750):
    plt.contourf(r_arr,z_arr,brt_arr[:,:,i])
    plt.contour(r_arr,z_arr,z[:,:,i])
    plt.plot(new_dict["R_LCFS"], new_dict["Z_LCFS"])
    plt.plot(new_dict["R_LIM"], new_dict["Z_LIM"])
    plt.savefig(f"im{i}")
    plt.clf()

import imageio
with imageio.get_writer('Normalizedcontour_withLCFS.gif', mode='I') as writer:
    for i in range(750):
        image = imageio.imread(f"NormalizedContour_withLCFS\im{i}.png")
        writer.append_data(image)


model = load_model("LSTMtest2")

x = np.argmax(model.predict(test_x),-1)


IoU2(test_y.reshape(50000,64,32), x.reshape(50000,64,32),10)
accuracy(test_y.reshape(50000,64,32), x.reshape(50000,64,32),10)
metric = tf.keras.metrics.MeanIoU(num_classes=11)
print(metric(test_y.reshape(50000,64,32), x.reshape(50000,64,32)))
model.evaluate(test_y,x, batch_size = 8)
y = np.argmax(model.predict(z.reshape(1,100,64,32)),-1)
mask = np.zeros(y.shape)

fig, ax = plt.subplots()

img1 = ax.imshow(brt[70])
img2 = ax.imshow(y[0,70], alpha=0.4, cmap="cool")

plt.show()

plt.imshow(y[0,70])
plt.show()

plt.imshow(z[70])
plt.show()


model = load_model("LSTMtest1")

im = x[10,70,:,:]

plt.imshow(test_y[10,70,:,:])
plt.show()

plt.imshow(im)
plt.show()

gif(x[0])

from Code.LSTM.utils import normalize, rightsize
import pickle

infile = open("CMod_data.pickle",'rb')
new_dict = pickle.load(infile)
infile.close()

brt_arr = new_dict["brt_arr"][:,:,0:100]

brt = rightsize(brt_arr)

z = normalize(brt_arr, normal="gauss",cutoff=20)

z = z/np.max(z)

y = np.argmax(model.predict(z.reshape(1,100,64,32)),-1)

y.shape

plt.imshow(z[20])
plt.show()

plt.imshow(y[0,20])
plt.show()

import cv2
import numpy as np

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

mask = np.zeros(y.shape)

fig, ax = plt.subplots()

img1 = ax.imshow(brt[10])
img2 = ax.imshow(y[0,10], alpha=0.4, cmap="cool")

plt.show()

mask = overlay_transparent(brt[10], y[0,10], 0,0)


values = ['1', '2', '3']

with open("file.txt", "w") as output:
    output.write(str(values))


