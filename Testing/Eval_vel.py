import numpy as np
import xarray as xr

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

vel_x = []
vel_y = []
for j in range(10):
    vel_x1 = []
    vel_y1 = []
    for i in range(100):
        data = xr.open_dataset(fr"Velocity_test/Velocity_{j+1}_{i}.nc")
        vel_x1.append(data["n"])
        vel_y1.append(data["blob_labels"])
    vel_x.append(vel_x1)
    vel_y.append(vel_y1)

with open("Pred_velx.npy", "rb") as f:
    x = np.load(f)

vel_y = np.moveaxis(np.array(vel_y),4,2)

x = x.reshape(10,100,64,64,32,21)

x1 = np.argmax(x,-1).astype("float64")

x1[np.where(x1!=0)] = 1

vel_y[np.where(vel_y!=0)] = 1

accuracy = []
conf = []
iou = []

for i in range(10):
    pred = x1[i].reshape(-1)
    label = vel_y[i].reshape(-1)
    acc = accuracy_score(label, pred)
    mat = confusion_matrix(label,pred)
    accuracy.append(acc)
    conf.append(mat)
    iou.append(mat[1,1]/(mat[0,1]+mat[1,0]+mat[1,1]))


accuracy
iou
conf[2]

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