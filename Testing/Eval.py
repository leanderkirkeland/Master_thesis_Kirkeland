import numpy as np
import xarray as xr
import tensorflow as tf

vel_x1 = []

for i in range(20):
    for j in range(100):
        data = xr.open_dataset(fr"Intermittency/Intermittency_n_{i}_{j}.nc")
        vel_x1.append(data["n"])

x1 = np.moveaxis(np.array(vel_x1),3,1).reshape(2000,64,64,32,1)
x1 = x1/x1.max()

vel_x2 = []

for i in range(20):
    for j in range(100):
        data = xr.open_dataset(fr"Intermittency/Intermittency_l_{i}_{j}.nc")
        vel_x2.append(data["n"])

x2 = np.moveaxis(np.array(vel_x2),3,1).reshape(2000,64,64,32,1)
x2 = x2/x2.max()

vel_x3 = []

for i in range(20):
    for j in range(100):
        data = xr.open_dataset(fr"Intermittency/Intermittency_h_{i}_{j}.nc")
        vel_x3.append(data["n"])

x3 = np.moveaxis(np.array(vel_x3),3,1).reshape(2000,64,64,32,1)
x3 = x3/x3.max()

model = tf.keras.models.load_model("Transformer")

pred1 = []
for i in range(20):
    pred = model.predict(x1[100*i:100*(i+1)])
    pred1.append(pred)

pred2 = []
for i in range(20):
    pred = model.predict(x2[100*i:100*(i+1)])
    pred2.append(pred)

pred3 = []
for i in range(20):
    pred = model.predict(x3[100*i:100*(i+1)])
    pred3.append(pred)

pred1 = np.array(pred1).reshape(2000,64,64,32,21)
pred2 = np.array(pred1).reshape(2000,64,64,32,21)
pred3 = np.array(pred1).reshape(2000,64,64,32,21)

with open("Pred_int_n.npy", "wb") as f:
    np.save(f, pred1)

with open("Pred_int_l.npy", "wb") as f:
    np.save(f, pred2)

with open("Pred_int_h.npy", "wb") as f:
    np.save(f, pred3)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

vel_y = []
for j in range(20):
    vel_x1 = []
    vel_y1 = []
    for i in range(100):
        data = xr.open_dataset(fr"Amp/Amp_vh_{j}_{i}.nc")
        vel_y1.append(data["blob_labels"])
    vel_y.append(vel_y1)

with open("Pred_Amp_vh.npy", "rb") as f:
    x = np.load(f)

vel_y = np.moveaxis(np.array(vel_y),4,2)

x = x.reshape(20,100,64,64,32,21)

x1 = np.argmax(x,-1).astype("float64")

x1[np.where(x1!=0)] = 1

vel_y[np.where(vel_y!=0)] = 1

accuracy = []
conf = []
iou = []

for i in range(20):
    pred = x1[i].reshape(-1)
    label = vel_y[i].reshape(-1)
    acc = accuracy_score(label, pred)
    mat = confusion_matrix(label,pred)
    accuracy.append(acc)
    conf.append(mat)
    iou.append(mat[1,1]/(mat[0,1]+mat[1,0]+mat[1,1]))

measures = np.array([accuracy, iou, conf], dtype=object)

with open("Pred_Amp_vh_measures.npy", "wb") as f:
    np.save(f, measures)

with open("Pred_both_h_measures.npy", "rb") as f:
    meas1 = np.load(f, allow_pickle=True)

with open("Pred_both_l_measures.npy", "rb") as f:
    meas2 = np.load(f, allow_pickle=True)

with open("Pred_int_n_measures.npy", "rb") as f:
    meas3 = np.load(f, allow_pickle=True)

with open("Pred_both_vh_measures.npy", "rb") as f:
    meas4 = np.load(f, allow_pickle=True)

plt.plot(np.arange(1,21),meas4[0], label="Sigma = 5")
plt.plot(np.arange(1,21),meas1[0], label="Sigma = 2")
plt.plot(np.arange(1,21),meas2[0], label="Sigma = 1")
plt.plot(np.arange(1,21),meas3[0], label="Sigma = 0")
plt.title("Accuracy with varying velocities and size")
plt.xlabel("Number of blobs")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1,21))
plt.legend()
plt.show()

plt.plot(np.arange(1,21),meas4[1], label="Sigma = 5")
plt.plot(np.arange(1,21),meas1[1], label="Sigma = 2")
plt.plot(np.arange(1,21),meas2[1], label="Sigma = 1")
plt.plot(np.arange(1,21),meas3[1], label="Sigma = 0")
plt.title("IoU with varying velocities and size")
plt.xlabel("Number of blobs")
plt.ylabel("IoU")
plt.xticks(np.arange(1,21))
plt.legend()
plt.show()

with open("Pred_Amp_vh_measures.npy", "rb") as f:
    meas4 = np.load(f, allow_pickle=True)

plt.plot(np.arange(1,21),meas4[0], label="Sigma = 20")
plt.plot(np.arange(1,21),meas3[0], label="Sigma = 0")
plt.title("Accuracy with varying amplitudes")
plt.xlabel("Number of blobs")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1,21))
plt.legend()
plt.show()

plt.plot(np.arange(1,21),meas4[1], label="Sigma = 20")
plt.plot(np.arange(1,21),meas3[1], label="Sigma = 0")
plt.title("IoU with varying amplitudes")
plt.xlabel("Number of blobs")
plt.ylabel("IoU")
plt.xticks(np.arange(1,21))
plt.legend()
plt.show()

plt.plot(np.arange(1,11),accuracy)
plt.xticks(np.arange(1,11))
plt.title("Accuracy with")
plt.xlabel("X-velocity")
plt.ylabel("Accuracy")
plt.show()

plt.plot(np.arange(1,11),iou)
plt.xticks(np.arange(1,11))
plt.title("IoU depending on x-velocity")
plt.xlabel("X-velocity")
plt.ylabel("IoU")
plt.show()

plt.imshow(x1[1,50,14])
plt.show()

plt.imshow(vel_y[1,50,14])
plt.show()

np.where(vel_y[4,50] == 1)




vel_y = []
for j in range(10):
    vel_y1 = []
    for i in range(100):
        data = xr.open_dataset(fr"Velocity_test/Velocity_{j+1}_{i}.nc")
        vel_y1.append(data["blob_labels"])
    vel_y.append(vel_y1)

with open("Pred_velx.npy", "rb") as f:
    x = np.load(f)
x = x.reshape(10,100,64,64,32,21)

x1 = np.argmax(x,-1)

vel_y = np.moveaxis(np.array(vel_y),4,2)

vel = np.zeros_like(x)
x_mul = np.zeros_like(x)


for i in range(20):
    vel[:,:,:,:,:,i+1][np.where(vel_y==i+1)] = 1
    x_mul[:,:,:,:,:,i+1][np.where(x1==i+1)] = 1


whole_blobs_label = []
whole_blobs_pred = []

for i in range(10):
    veli = []
    velx = []
    for j in range(100):
        samp = []
        sampx = []
        for k in range(21):
            if vel[i,j,:,:,:,k].max() == 1.0:
                samp.append(vel[i,j,:,:,:,k])
            if x_mul[i,j,:,:,:,k].max() == 1.0:
                sampx.append(x_mul[i,j,:,:,:,k])
        veli.append(samp)
        velx.append(sampx)
    whole_blobs_label.append(veli)
    whole_blobs_pred.append(velx)

best_matches = []

for i in range(10):
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


accuracy = []
iou = []


for i in range(10):
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
    
import matplotlib.pyplot as plt

plt.plot(np.arange(1,11),accuracy)
plt.xticks(np.arange(1,11))
plt.title("Accuracy depending on x-velocity")
plt.xlabel("X-velocity")
plt.ylabel("Accuracy")
plt.show()

plt.plot(np.arange(1,11),iou)
plt.xticks(np.arange(1,11))
plt.title("IoU depending on x-velocity")
plt.xlabel("X-velocity")
plt.ylabel("IoU")
plt.show()

x = np.random.normal(2,2, size=2000)
np.where(x<0.1,0.1,x )