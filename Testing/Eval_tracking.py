import numpy as np
import xarray as xr
import tensorflow as tf

vel_x1 = []
vel_y1 = []

for i in range(100):
    data = xr.open_dataset(fr"Tracking/Tracking_{i}.nc")
    vel_x1.append(data["n"])
    vel_y1.append(data["blob_labels"])

x1 = np.moveaxis(np.array(vel_x1),3,1).reshape(100,64,64,32,1)
y1 = np.moveaxis(np.array(vel_y1),3,1).reshape(100,64,64,32,1)
x1 = x1/x1.max()

model = tf.keras.models.load_model("Transformer")

pred = model.predict(x1)

with open("Tracking.npy", "wb") as f:
    np.save(f, pred)

with open("Tracking.npy", "rb") as f:
    track = np.load(f)

import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy
import shapely
import cv2
import skimage
import scipy
from shapely.geometry import Polygon
import math


track1 = np.argmax(track,-1)
count1 = 0
for i in range(100):
    if i > 0:
        count1 += len(np.unique(track1[i-1]))-1
        track1[i][np.where(track1[i] != 0)] += count1

count = 0
for i in range(100):
    if i > 0:
        count += len(np.unique(y1[i-1]))-1
        y1[i][np.where(y1[i] != 0)] += count

track1.max()

track1 = track1.reshape(6400,64,32)
y1 = y1.reshape(6400,64,32)


lifetimet = []
track11 = track1.copy()
comt = []
labels = []
for i in np.unique(track1):
    lifetimet.append(len(np.unique(np.where(track1==i+1)[0])))
    com1 = []
    for j in np.unique(np.where(track1==i+1)[0]):
        track11[j][np.where(track1[j]==i+1)] = x1.reshape([6400,64,32])[j][np.where(track1[j]==i+1)]
        u = scipy.ndimage.center_of_mass(track11[j])
        if math.isnan(u[0]):
            None
        else:
            com1.append(u)
    labels.append(i)
    comt.append(com1)



y1 = y1.astype(int)

lifetimel = []
y11 = y1.copy()
coml = []
labelsy = []
for i in np.unique(y1)-1:
    lifetimel.append(len(np.unique(np.where(y1==i+1)[0])))
    com1 = []
    if lifetimel[i] >= 2:
        for j in np.unique(np.where(y1==i+1)[0]):
            y11[j][np.where(y1[j]==i+1)] = x1.reshape([6400,64,32])[j][np.where(y1[j]==i+1)]
            u1 = scipy.ndimage.center_of_mass(y11[j])
            if math.isnan(u1[0]):
                None
            else:
                com1.append(u1)
        labelsy.append(i)
        coml.append(com1)

x_velocityt = []
y_velocityt = []

for i in range(len(comt)):
    x_vel = 0
    y_vel = 0
    if len(comt[i])>2:
        for j in range(len(comt[i])-1):
            y_vel += comt[i][j+1][0]-comt[i][j][0]
            x_vel += comt[i][j+1][1]-comt[i][j][1]
        x_velocityt.append(x_vel/len(comt[i]))
        y_velocityt.append(y_vel/len(comt[i]))

x_velocityl = []
y_velocityl = []

for i in range(len(coml)):
    x_vel = 0
    y_vel = 0
    if len(coml[i])>2:
        for j in range(len(coml[i])-1):
            y_vel += coml[i][j+1][0]-coml[i][j][0]
            x_vel += coml[i][j+1][1]-coml[i][j][1]
        x_velocityl.append(x_vel/len(coml[i]))
        y_velocityl.append(y_vel/len(coml[i]))

np.min(x_velocityl)
np.min(x_velocityt)

plt.hist([y_velocityl,y_velocityt], label = ["Labels","Tracking"])
plt.title("Distribution of y-velocities")
plt.xlabel("Center of mass y-velocity/pixels/frame")
plt.ylabel("Number of blobs")
plt.legend()
plt.show()
plt.hist([x_velocityl,x_velocityt],label = ["Labels","Tracking"])
plt.title("Distribution of x-velocities")
plt.xlabel("Center of mass x-velocity/pixels/frame")
plt.ylabel("Number of blobs")
plt.legend()
plt.show()

np.min(y_velocityt)
np.min(y_velocityl)

np.where(x_velocity)

np.argsort(x_velocity)

np.where(track1 == 483)

for i in range(8):
    plt.imshow(track1[2523+i])
    plt.show()


abslot = []

for i in range(len(comt)-1):
    if np.where(track1 == i+1)[0].size > 0:
        abslot.append(x1.reshape(6400,64,32)[np.where(track1 == i+1)].max())

abslol = []

for i in range(len(coml)-1):
    abslol.append(x1.reshape(6400,64,32)[np.where(y1 == i+1)].max())

np.mean(abslol)
np.mean(abslot)

plt.hist([abslol,abslot],label = ["Labels","Tracking"])
plt.title("Distribution of amplitdues")
plt.xlabel("Max amplitdue in blob/counts")
plt.ylabel("Number of blobs")
plt.legend()
plt.show()

np.min(abslot)
np.min(abslol)

countt = []

for i in range(np.max(track1)-1):
    if np.where(track1 == i+1)[0].size > 0:
        count = []
        for j in np.unique(np.where(track1==i+1)[0]):
            count.append(np.count_nonzero(track1[j] == i+1))
        countt.append(np.max(count))

countl = []

for i in range(np.max(y1)-1):
    if np.where(y1 == i+1)[0].size > 0:
        count = []
        for j in np.unique(np.where(y1==i+1)[0]):
            count.append(np.count_nonzero(y1[j] == i+1))
        countl.append(np.max(count))


plt.hist([countl,countt],label = ["Labels","Tracking"])
plt.title("Distribution of sizes")
plt.xlabel("Maximum size of blobs/pixels")
plt.ylabel("Number of blobs")
plt.legend()
plt.show()

np.max(countl)
np.max(countt)

for i in range(np.max(y1)-1):
    
track1[1].count(1)

np.count_nonzero(track1 == 1)