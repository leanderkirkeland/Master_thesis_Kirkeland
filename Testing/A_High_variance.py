import numpy as np
import xarray as xr
import tensorflow as tf

model = tf.keras.models.load_model("Transformer")

vel_x1 = []

for i in range(20):
    for j in range(100):
        data = xr.open_dataset(fr"High_variance/Vel_vh_{i}_{j}.nc")
        vel_x1.append(data["n"])

x1 = np.moveaxis(np.array(vel_x1),3,1).reshape(2000,64,64,32,1)
x1 = x1/x1.max()

pred1 = []
for i in range(20):
    pred = model.predict(x1[100*i:100*(i+1)])
    pred1.append(pred)

pred1 = np.array(pred1).reshape(2000,64,64,32,21)

with open("Pred_vel_vh.npy", "wb") as f:
    np.save(f, pred1)

vel_x1 = []

for i in range(20):
    for j in range(100):
        data = xr.open_dataset(fr"High_variance/Size_vh_{i}_{j}.nc")
        vel_x1.append(data["n"])

x1 = np.moveaxis(np.array(vel_x1),3,1).reshape(2000,64,64,32,1)
x1 = x1/x1.max()

pred1 = []
for i in range(20):
    pred = model.predict(x1[100*i:100*(i+1)])
    pred1.append(pred)

pred1 = np.array(pred1).reshape(2000,64,64,32,21)

with open("Pred_Size_vh.npy", "wb") as f:
    np.save(f, pred1)

vel_x1 = []

for i in range(20):
    for j in range(100):
        data = xr.open_dataset(fr"High_variance/Both_vh_{i}_{j}.nc")
        vel_x1.append(data["n"])

x1 = np.moveaxis(np.array(vel_x1),3,1).reshape(2000,64,64,32,1)
x1 = x1/x1.max()

pred1 = []
for i in range(20):
    pred = model.predict(x1[100*i:100*(i+1)])
    pred1.append(pred)

pred1 = np.array(pred1).reshape(2000,64,64,32,21)

with open("Pred_Both_vh.npy", "wb") as f:
    np.save(f, pred1)