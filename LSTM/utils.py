import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

def rightsize(dataset):
    if dataset.shape[1] == 64 and dataset.shape[0] == 64:
        dataset = np.moveaxis(np.moveaxis(np.flip(np.rot90(dataset[:32,:,:], k=1), axis = 1),1,2),0,1)
    elif dataset.shape[1] == 64 and dataset.shape[2] == 32:
        None
    else:
        raise Exception("Array should be of size (N, 64, 32)")
    return dataset

def normalize(dataset, cutoff = 15, normal = None):
    if normal == None:
        dataset = rightsize(dataset)
        arr = np.zeros(dataset.shape)
        for i in range(dataset.shape[0]):
            arr[i] = dataset[i] - np.mean(dataset, axis=0)
        cut = np.where(arr < cutoff, 0, arr)
    elif normal == "med":
        dataset = rightsize(dataset)
        arr = np.zeros(dataset.shape)
        for i in range(dataset.shape[0]):
            arr[i] = dataset[i] - np.mean(dataset, axis=0)
        arr = ndimage.median_filter(arr, 3)
        cut = np.where(arr < cutoff, 0, arr)
    elif normal == "gauss":
        #dataset = rightsize(dataset)
        arr = np.zeros(dataset.shape)
        for i in range(dataset.shape[0]):
            arr[i,:,:] = dataset[i,:,:] - np.mean(dataset, axis=0)
        arr = ndimage.gaussian_filter(arr, 2)
        cut = np.where(arr < cutoff, 0, arr)
    return cut

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

import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
with open("1150709030_short_raw.pickle","rb") as f:
    rawdata = pickle.load(f)

p = np.flip(np.moveaxis(np.rot90(rawdata["brt_arr"][:32,:,:]),2,0),axis = 2)


plt.imshow(rawdata["brt_arr"][:,:,1000])
plt.show()
plt.imshow(p[1000])
plt.show()

processes50 = normalize(p, normal = "gauss", cutoff=50)
processes100 = normalize(p, normal = "gauss", cutoff=100)
processes150 = normalize(p, normal = "gauss", cutoff=150)

exp50 = np.zeros((32,64,64,32,1))
exp100 = np.zeros((32,64,64,32,1))
exp150 = np.zeros((32,64,64,32,1))



for i in range(31):
    exp50[i] = processes50[i*64:(i+1)*64].reshape(64,64,32,1)
    exp100[i] = processes100[i*64:(i+1)*64].reshape(64,64,32,1)
    exp150[i] = processes150[i*64:(i+1)*64].reshape(64,64,32,1)

exp50[-1][0:16] = processes50[1984:].reshape(16,64,32,1)
exp100[-1][0:16] = processes100[1984:].reshape(16,64,32,1)
exp150[-1][0:16] = processes150[1984:].reshape(16,64,32,1)

exp50[3].max()

for i in range(32):
    if exp50[i].max() != 0:
        exp50[i] = exp50[i]/exp50[i].max()
    if exp100[i].max() != 0:
        exp100[i] = exp100[i]/exp100[i].max()
    if exp150[i].max() != 0:
        exp150[i] = exp150[i]/exp150[i].max()

with open("Experimental_50.npy","wb") as f:
    np.save(f, exp50)

with open("Experimental_100.npy","wb") as f:
    np.save(f, exp100)

with open("Experimental_150.npy","wb") as f:
    np.save(f, exp150)

"""

gif(rightsize(brt_arr), name="org")

gif(normalize(brt_arr, normal="med"), name = "Normalmed15")

gif(normalize(brt_arr, normal="gauss"), name = "Normalgauss15")

gif(normalize(brt_arr), name = "Normalized15")

gif(normalize(brt_arr, normal="med",cutoff=20), name = "Normalmed20")

gif(normalize(brt_arr, normal="gauss",cutoff=20), name = "Normalgauss20")

gif(normalize(brt_arr, cutoff=20), name = "Normalized20")

import pickle

infile = open("CMod_data.pickle",'rb')
new_dict = pickle.load(infile)
infile.close()

brt_arr = new_dict["brt_arr"][:,:,0:100]

brt_arr.shape

import matplotlib.pyplot as plt

normalize(brt_arr)

plt.imshow(normalize(brt_arr)[100])
plt.show()

plt.imshow(rightsize(brt_arr)[100])
plt.show()
plt.imshow(brt_arr[:,:,100])
plt.show()

"""
