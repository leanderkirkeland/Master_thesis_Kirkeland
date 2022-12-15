import tensorflow as tf
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

intermittency_x = []
intermittency_y = []
for j in range(20):
    intermittency_x1 = []
    intermittency_y1 = []
    for i in range(5):
        data = xr.open_dataset(fr"Intermittency_test/intermittency_{j}_{i}.nc")
        intermittency_x1.append(data["n"])
        intermittency_y1.append(data["blob_labels"])
    intermittency_x.append(intermittency_x1)
    intermittency_y.append(intermittency_y1)


x = np.moveaxis(np.array(intermittency_x).reshape(100,64,32,64,1),4,2)
x = x/x.max()
y = np.moveaxis(np.array(intermittency_y).reshape(100,64,32,64,1),3,1)

model = tf.keras.models.load_model("Transformer")

pred = model.predict(x)

with open("Pred_inter.npy", "wb") as f:
    np.save(f, pred)

with open("Intermittency_test/Pred_inter.npy", "rb") as f:
    x = np.load(f)


x = np.argmax(x,-1)
y = y.reshape(100,64,64,32)



for j in range(100):
    for k in range(20):
        count = 0
        for i in range(64):
            if k in x[j,i]:
                count+=1
                if np.count_nonzero(x[j,i] == k+1) < 10:
                    x[j,i][x[j,i] == k+1] = 0
        if count < 5:
            x[j][x[j] == k] = 0

count = []
mean = []
counter = 0
for i in range(20):
    c1 = []
    for j in range(5):
        c1.append(np.unique(x[counter]).shape[0]-1)
        counter += 1
    count.append(c1)
    mean.append(np.mean(c1)/(i+1))

plt.plot(mean)
plt.show()

plt.imshow(y[90,55])
plt.show()
plt.imshow(x[90,55])
plt.show()

np.count_nonzero(x[0,16] == 1)


intermittency_x = []
intermittency_y = []
for j in range(20):
    intermittency_x1 = []
    intermittency_y1 = []
    for i in range(5):
        data = xr.open_dataset(fr"Intermittency_large/Intermittency_{j}_{i}.nc")
        intermittency_x1.append(data["n"])
        intermittency_y1.append(data["blob_labels"])
    intermittency_x.append(intermittency_x1)
    intermittency_y.append(intermittency_y1)

with open("Intermittency_large/Pred_inter.npy", "rb") as f:
    x = np.load(f)

x = np.argmax(x,-1)
y = y.reshape(100,64,64,32)

for j in range(2000):
    for k in range(20):
        count = 0
        for i in range(64):
            if k in x[j,i]:
                count+=1
                if np.count_nonzero(x[j,i] == k+1) < 15:
                    x[j,i][x[j,i] == k+1] = 0
        if len(np.unique(np.where(x[i] == k+1)[0])) > 3:
            x[j][x[j] == k] = 0



count = []
mean = []
counter = 0
for i in range(20):
    c1 = []
    for j in range(100):
        c1.append(np.unique(x[counter]).shape[0]-1)
        counter += 1
    count.append(c1)
    mean.append(np.mean(c1)/(i+1))

count[2]
mean

print()

plt.plot(np.arange(20)+1,mean)
plt.title("Intermittency of blobs")
plt.xlabel("Number of blobs in dataset")
plt.ylabel("Percentage of blobs found")
plt.show()

np.where(y==1)

plt.imshow(y[11,32])
plt.show()

np.unique()

plt.imshow(x[150,32])
plt.show()

import bz2
import pickle

data = bz2.BZ2File(r"C:\Users\Leander\Skole\H2022\short_1.21_processed.pbz2", "rb")
data = pickle.load(data)

data["brt_true"].shape


vel_x = []

for i in range(10):
    for j in range(100):
        data = xr.open_dataset(fr"Velocity_test/Velocity_{i+1}_{j}.nc")
        vel_x.append(data["n"])

x = np.moveaxis(np.array(vel_x),3,1).reshape(1000,64,64,32,1)
x = x/x.max()

model = tf.keras.models.load_model("Transformer")

pred = model.predict(x)

with open("Pred_velx.npy", "wb") as f:
    np.save(f, pred)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

intermittency_x = []
intermittency_y = []
for j in range(20):
    intermittency_x1 = []
    intermittency_y1 = []
    for i in range(100):
        data = xr.open_dataset(fr"Intermittency_large/Intermittency_{j}_{i}.nc")
        intermittency_x1.append(data["n"])
        intermittency_y1.append(data["blob_labels"])
    intermittency_x.append(intermittency_x1)
    intermittency_y.append(intermittency_y1)

with open("Intermittency_large/Pred_inter.npy", "rb") as f:
    x = np.load(f)

intermittency_y = np.moveaxis(np.array(intermittency_y),4,2)

x = x.reshape(20,100,64,64,32,21)

x1 = np.argmax(x,-1).astype("float64")

x1[np.where(x1!=0)] = 1

intermittency_y[np.where(intermittency_y!=0)] = 1

accuracy = []
conf = []
iou = []

for i in range(20):
    pred = x1[i].reshape(-1)
    label = intermittency_y[i].reshape(-1)
    acc = accuracy_score(label, pred)
    mat = confusion_matrix(label,pred)
    accuracy.append(acc)
    conf.append(mat)
    iou.append(mat[1,1]/(mat[0,1]+mat[1,0]+mat[1,1]))

iou



plt.plot(np.arange(1,21),accuracy)
plt.xticks(np.arange(1,21))
plt.title("Accuracy depending on number of blobs")
plt.xlabel("Number of blobs")
plt.ylabel("Accuracy")
plt.show()

plt.plot(np.arange(1,21),iou)
plt.xticks(np.arange(1,21))
plt.title("IoU depending on number of blobs")
plt.xlabel("Numbers of blobs")
plt.ylabel("IoU")
plt.show()

