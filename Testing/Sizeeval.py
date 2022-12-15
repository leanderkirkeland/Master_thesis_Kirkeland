import numpy as np
import tensorflow as tf
import xarray as xr

size_x = []

for i in range(15):
    for j in range(100):
        data = xr.open_dataset(fr"Size_test/Size_{i}_{j}.nc")
        size_x.append(data["n"])

x = np.moveaxis(np.array(size_x),3,1).reshape(1500,64,64,32,1)
x = x/x.max()

model = tf.keras.models.load_model("Transformer")

pred = model.predict(x)

with open("Pred_size.npy", "wb") as f:
    np.save(f, pred)


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

size_y = []
for j in range(15):
    size_y1 = []
    for i in range(100):
        data = xr.open_dataset(fr"Size_test/Size_{j}_{i}.nc")
        size_y1.append(data["blob_labels"])
    size_y.append(size_y1)

with open("Pred_size.npy", "rb") as f:
    x = np.load(f)

size_y = np.moveaxis(np.array(size_y),4,2)

x = x.reshape(15,100,64,64,32,21)

x1 = np.argmax(x,-1).astype("float64")

x1[np.where(x1!=0)] = 1

size_y[np.where(size_y!=0)] = 1

accuracy = []
conf = []
iou = []

for i in range(15):
    pred = x1[i].reshape(-1)
    label = size_y[i].reshape(-1)
    acc = accuracy_score(label, pred)
    mat = confusion_matrix(label,pred)
    accuracy.append(acc)
    conf.append(mat)
    iou.append(mat[1,1]/(mat[0,1]+mat[1,0]+mat[1,1]))

accuracy
conf[10]

plt.plot(np.arange(2,32,2),accuracy)
plt.xticks(np.arange(2,32,2))
plt.title("Accuracy depending on size")
plt.xlabel("Size")
plt.ylabel("Accuracy")
plt.show()

plt.plot(np.arange(2,32,2),iou)
plt.xticks(np.arange(2,32,2))
plt.title("IoU depending on size")
plt.xlabel("Size")
plt.ylabel("IoU")
plt.show()

plt.imshow(x1[14,50,32])
plt.show()

plt.imshow(size_y[14,50,32])
plt.show()


size_y = []
for j in range(15):
    size_y1 = []
    for i in range(100):
        data = xr.open_dataset(fr"Size_test/size_{j}_{i}.nc")
        size_y1.append(data["blob_labels"])
    size_y.append(size_y1)

with open("Pred_size.npy", "rb") as f:
    x = np.load(f)
x = x.reshape(15,100,64,64,32,21)

x1 = np.argmax(x,-1)

size_y = np.moveaxis(np.array(size_y),4,2)

size = np.zeros_like(x)
x_mul = np.zeros_like(x)


for i in range(20):
    size[:,:,:,:,:,i+1][np.where(size_y==i+1)] = 1
    x_mul[:,:,:,:,:,i+1][np.where(x1==i+1)] = 1


whole_blobs_label = []
whole_blobs_pred = []

for i in range(15):
    sizei = []
    sizex = []
    for j in range(100):
        samp = []
        sampx = []
        for k in range(21):
            if size[i,j,:,:,:,k].max() == 1.0:
                samp.append(size[i,j,:,:,:,k])
            if x_mul[i,j,:,:,:,k].max() == 1.0:
                sampx.append(x_mul[i,j,:,:,:,k])
        sizei.append(samp)
        sizex.append(sampx)
    whole_blobs_label.append(sizei)
    whole_blobs_pred.append(sizex)

best_matches = []

for i in range(15):
    best_all = []
    for j in range(100):
        best = []
        for k in range(len(whole_blobs_label[i][j])):
            acc = 0
            for p in range(len(whole_blobs_pred[i][j])):
                acc1 = accuracy_score(np.array(whole_blobs_label[i][j][k]).reshape(-1), np.array(whole_blobs_pred[i][j][p]).reshape(-1))
                if acc1 > acc:
                    prelab = p
                    acc = acc1
            acc = 0
            best.append(prelab)
        best_all.append(best)
    best_matches.append(best_all)


best_matches[14][0][5]

accuracy = []
iou = []

np.array(whole_blobs_pred[14][1][0])[60].max()

plt.imshow(np.array(whole_blobs_label[14][1][1])[60])
plt.show()
plt.imshow(np.array(whole_blobs_pred[14][1][best_matches[14][0][1]])[60])
plt.show()

plt.imshow(size_y[14,1,60])
plt.show()
plt.imshow(x1[14,1,60])
plt.show()
plt.imshow(size_x[14,1,60])
plt.show()

size_x = []
for j in range(15):
    size_x1 = []
    for i in range(100):
        data = xr.open_dataset(fr"Size_test/size_{j}_{i}.nc")
        size_x1.append(data["n"])
    size_x.append(size_x1)
size_x = np.moveaxis(np.array(size_x),4,2)

for i in range(15):
    counter = 0
    accer = 0
    iou1 = 0
    for j in range(100):
        for k in range(len(whole_blobs_label[i][j])):
            accer += accuracy_score(np.array(whole_blobs_label[i][j][k]).reshape(-1), np.array(whole_blobs_pred[i][j][best_matches[i][j][k]]).reshape(-1))
            conf1 = confusion_matrix(np.array(whole_blobs_label[i][j][k]).reshape(-1),np.array(whole_blobs_pred[i][j][best_matches[i][j][k]]).reshape(-1))
            iou1 += conf1[1,1]/(conf1[0,1]+conf1[1,0]+conf1[1,1])
            counter +=1
    accuracy.append(accer/counter)
    iou.append(iou1/counter)

plt.plot(np.arange(2,32,2),accuracy)
plt.xticks(np.arange(2,32,2))
plt.title("Accuracy depending on size")
plt.xlabel("Size")
plt.ylabel("Accuracy")
plt.show()

plt.plot(np.arange(2,32,2),iou)
plt.xticks(np.arange(2,32,2))
plt.title("IoU depending on size")
plt.xlabel("Size")
plt.ylabel("IoU")
plt.show()