import numpy as np
import xarray as xr
import tensorflow as tf

vel_x1 = []
vel_y1 = []

for i in range(20):
    for j in range(100):
        data = xr.open_dataset(fr"Amp/Amp_vh_{i}_{j}.nc")
        vel_x1.append(data["n"])
        vel_y1.append(data["blob_labels"])

x1 = np.moveaxis(np.array(vel_x1),3,1).reshape(2000,64,64,32,1)
y1 = np.moveaxis(np.array(vel_y1),3,1).reshape(2000,64,64,32,1)
x1 = x1/x1.max()

plt.imshow(y1[600,32,:,:,0])
plt.show()

model = tf.keras.models.load_model("Transformer")

pred1 = []
for i in range(20):
    pred = model.predict(x1[100*i:100*(i+1)])
    pred1.append(pred)

pred1 = np.array(pred1).reshape(2000,64,64,32,21)

with open("Pred_Amp_vh.npy", "wb") as f:
    np.save(f, pred1)