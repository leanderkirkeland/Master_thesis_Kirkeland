import numpy as np
import xarray as xr
import tensorflow as tf

vel_x1 = []

for i in range(20):
    for j in range(100):
        data = xr.open_dataset(fr"Intermittency/Intermittency_h_{i}_{j}.nc")
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

model = tf.keras.models.load_model("Transformer")

pred1 = []
for i in range(20):
    pred = model.predict(x1[100*i:100*(i+1)])
    pred1.append(pred)

pred2 = []
for i in range(20):
    pred = model.predict(x2[100*i:100*(i+1)])
    pred2.append(pred)

pred1 = np.array(pred1).reshape(2000,64,64,32,21)
pred2 = np.array(pred2).reshape(2000,64,64,32,21)

with open("Pred_int_h.npy", "wb") as f:
    np.save(f, pred1)

with open("Pred_int_l.npy", "wb") as f:
    np.save(f, pred2)


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

vel_y = []
for j in range(20):
    vel_y1 = []
    for i in range(100):
        data = xr.open_dataset(fr"Vel/Vel_h_{j}_{i}.nc")
        vel_y1.append(data["blob_labels"])
    vel_y.append(vel_y1)

with open("Pred_vel_h.npy", "rb") as f:
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

with open("Pred_vel_h_measures.npy", "wb") as f:
    np.save(f, measures)

with open("Pred_vel_h_measures.npy", "rb") as f:
    meas1 = np.load(f, allow_pickle=True)

with open("Pred_vel_l_measures.npy", "rb") as f:
    meas2 = np.load(f, allow_pickle=True)

with open("Pred_int_n_measures.npy", "rb") as f:
    meas3 = np.load(f, allow_pickle=True)



plt.plot(np.arange(1,21),meas1[0], label="Sigma = 2")
plt.plot(np.arange(1,21),meas2[0], label="Sigma = 1")
plt.plot(np.arange(1,21),meas3[0], label="Sigma = 0")
plt.title("Accuracy with varying x-velocities")
plt.xlabel("Number of blobs")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1,21))
plt.legend()
plt.show()

plt.plot(np.arange(1,21),meas1[1], label="Sigma = 2")
plt.plot(np.arange(1,21),meas2[1], label="Sigma = 1")
plt.plot(np.arange(1,21),meas3[1], label="Sigma = 0")
plt.title("IoU with varying x-velocities")
plt.xlabel("Number of blobs")
plt.ylabel("IoU")
plt.xticks(np.arange(1,21))
plt.legend()
plt.show()