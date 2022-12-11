import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy
import shapely
import cv2
import skimage
import scipy
from shapely.geometry import Polygon

with open("EXP2\exp50_pred2.npy","rb") as f:
    x1 = np.load(f)

with open("EXP2\exp100_pred2.npy","rb") as f:
    x2 = np.load(f)

with open("EXP2\exp150_pred2.npy","rb") as f:
    x3 = np.load(f)

with open(r"C:\Users\Leander\Skole\H2022\1150709030_short_raw.pickle","rb") as f:
    raw = pickle.load(f)

with open("EXP2\Experimental_50.npy","rb") as f:
    expx1 = np.load(f)

with open("Exp_LSTM.npy","rb") as f:
    lstm_pred = np.load(f)

with open(r"C:\Users\Leander\Skole\H2022\1150709030_short_raft.pickle","rb") as f:
    raft = pickle.load(f)

data50 = np.argmax(x1,-1)
data100 = np.argmax(x2,-1)
data150 = np.argmax(x3,-1)
lstm_pred = np.argmax(lstm_pred,-1)

rawdata = np.flip(np.moveaxis(np.rot90(raw["brt_arr"][:32,:,:]),2,0),axis = 2)

orgdata50 = data50.copy()
orgdata100 = data100.copy()
orgdata150 = data150.copy()

data50  = data50.reshape(2048,64,32)
data100 = data100.reshape(2048,64,32)
data150 = data150.reshape(2048,64,32)
orgdata50  = orgdata50.reshape(2048,64,32)
orgdata100 = orgdata100.reshape(2048,64,32)
orgdata150 = orgdata150.reshape(2048,64,32)
expx1 = expx1.reshape(2048,64,32)
lstm_pred = lstm_pred.reshape(2000,64,32)
lstm_pred1 = lstm_pred.copy()

data50[np.where(expx1 == 0)] = 0
lstm_pred1[np.where(expx1[:2000] == 0)] = 0


label1, features1 = scipy.ndimage.label(np.where(data50 != 0,1,0))
label2, features2 = scipy.ndimage.label(np.where(lstm_pred1 != 0,1,0))
rawdata64 = np.flip(np.flip(np.moveaxis(np.rot90(raw["brt_arr"]),2,0),axis = 2),axis=1)


label1[np.where(expx1 == 0)] = 0
label2[np.where(expx1[:2000] == 0)] = 0

len(np.unique(label2))

q = np.unique(np.where(lstm_pred != 0)[0])

np.where(q == 1892)

for i in range(1):
    i = 1226
    plt.imshow(rawdata64[i,:,32:])
    for j in range(len(raft["output_tracking"][q[i]])):
        y,x = raft["output_tracking"][q[i]][j][5].exterior.xy
        x = np.array(x)*64
        y = np.array(y)*64-32
        plt.plot(y,x,"red")
    plt.plot(0,0,"Red", label = "RAFT")
    lab_test1 = skimage.measure.find_contours(np.flip(orgdata50[q[i]],axis=0))
    for j in range(len(lab_test1)):
        plt.plot(lab_test1[j][:,1],lab_test1[j][:,0],"blue")
    plt.plot(0,0,"blue", label = "Transformer")
    lab_test2 = skimage.measure.find_contours(np.flip(lstm_pred[q[i]],axis=0))
    for j in range(len(lab_test2)):
        plt.plot(lab_test2[j][:,1],lab_test2[j][:,0],"black")
    plt.plot(0,0,"black", label = "LSTM")
    plt.title(f"Blob Detection {q[i]}")
    plt.legend()
    plt.show()

rft = []
Trans = []
lsm = []

for i in range(2000):
    Trans1 = []
    lab_test1 = skimage.measure.find_contours(np.flip(orgdata50[i],axis=0))
    for j in range(len(lab_test1)):
        x2 = lab_test1[j][:,1]
        y2 = lab_test1[j][:,0]
        Trans1.append((x2,y2))
    Trans.append(Trans1)
    lsm1 = []
    lab_test1 = skimage.measure.find_contours(np.flip(lstm_pred[i],axis=0))
    for j in range(len(lab_test1)):
        x3 = lab_test1[j][:,1]
        y3 = lab_test1[j][:,0]
        lsm1.append((x3,y3))
    lsm.append(lsm1)
    rft1 = []
    for j in range(len(raft["output_tracking"][i])):
        x1,y1 = raft["output_tracking"][i][j][5].exterior.xy
        x1 = np.array(x1)*64
        y1 = np.array(y1)*64-32
        rft2 = []
        for o in range(len(x1)):
            rft2.append((x1[o],y1[o]))
        rft1.append(rft2)
    rft.append(rft1)




xpoly = []
ypoly = []
rpoly = []
for i in range(2000):
    xpoly1 = []
    for j in range(len(skimage.measure.find_contours(np.flip(orgdata50[i],axis=0)))):
        if len(skimage.measure.find_contours(np.flip(orgdata50[i],axis=0))[j]) > 2:
            xpoly1.append(Polygon(skimage.measure.find_contours(np.flip(orgdata50[i],axis=0))[j]))
    xpoly.append(xpoly1)
    ypoly1 = []
    for j in range(len(skimage.measure.find_contours(np.flip(lstm_pred[i],axis=0)))):
        if len(skimage.measure.find_contours(np.flip(lstm_pred[i],axis=0))[j]) > 2:
            ypoly1.append(Polygon(skimage.measure.find_contours(np.flip(lstm_pred[i],axis=0))[j]))
    ypoly.append(ypoly1)
    rpoly1 = []
    for j in range(len(rft[i])):
        rpoly1.append(Polygon(np.array(rft[i][j])))
    rpoly.append(rpoly1)

xpoly[1285][0].intersection(rpoly[1285][0]).area

transnomatchraft = 0
lstmnomatchraft = 0
transraft = []
lstmraft = []

for i in range(1999):
    transraft1= []
    for j in range(len(xpoly[i])):
        for k in range(len(rpoly[i])):
            if xpoly[i][j].intersection(rpoly[i][k]).area == 0.0:
                transnomatchraft += 1
            else:
                transraft1.append(xpoly[i][j].intersection(rpoly[i][k]).area/xpoly[i][j].union(rpoly[i][k]).area)
    transraft.append(transraft1)
    lstmraft1 = []
    for j in range(len(ypoly[i])):
        for k in range(len(rpoly[i])):
            if ypoly[i][j].intersection(rpoly[i][k]).area == 0.0:
                lstmnomatchraft += 1
            else:
                lstmraft1.append(ypoly[i][j].intersection(rpoly[i][k]).area/ypoly[i][j].union(rpoly[i][k]).area)
    lstmraft.append(lstmraft1)


np.max(np.sum(lstmraft))

np.argmax(lstmraft)
np.argmax(transraft)
for i in range(1999):
    transraft[i] = np.sum(transraft[i])
    lstmraft[i] = np.sum(lstmraft[i])

transraft
lstmraft

len(np.where(np.array(transraft) > 0.1)[0])
len(np.where(np.array(lstmraft) > 0.1)[0])


for i in range(len(q)):    
    plt.imshow(expx1[q][100+i])
    plt.title("Input")
    plt.show()
    plt.imshow(orgdata50[q][100+i])
    plt.title("Transformer")
    plt.show()
    plt.imshow(lstm_pred[q][100+i])
    plt.title("LSTM")
    plt.show()
    print(q[i+100])

counter = 0
for i in range(1999):
    if raft["output_tracking"][i] != []:
        counter += 1

for i in range(2000):
    plt.imshow(expx1[i])
    plt.show()
    plt.imshow(orgdata50[i])
    plt.show()
    print(i)

for i in range(2000):
    plt.imshow(orgdata50[i])
    plt.show()
    print(i)

plt.imshow(orgdata50[192])
plt.title("Output of the model")
plt.show()
plt.imshow(data50[192])
plt.title("Output after removing 'blobs' with less than 5 pixels and over 3 timesteps")
plt.show()

plt.contourf(rawdata[192])
x1 = plt.contour(label[192], name = "Blob 1")
plt.title("Removed output over original data")
plt.clabel(x1)
plt.show()
plt.imshow(np.flip(expx1[196],axis=0))
plt.title("Input of the model")
plt.show()

plt.imshow(expx1[192])
plt.imshow(label[192])
plt.show()

for i in range(2000):
    if raft["output_tracking"][i] != []:
        print(i,raft["output_tracking"][i][0][2], raft["output_tracking"][i][0][3])

blobsss = []
for i in range(1999):
    for j in range(len(raft["output_tracking"][i])):
        blobsss.append(raft["output_tracking"][i][j][0])

len(np.unique(blobsss))



x,y = raft["output_tracking"][1800][0][5].exterior.xy
xu, yu = raft["output_tracking"][439][1][4].exterior.xy

x = np.array(x)*64
y = np.array(y)*64-32
xu = np.array(xu)*64
yu = np.array(yu)*64-32


plt.imshow(rawdata64[446])
plt.plot(y,x)
plt.show()

plt.imshow(rawdata64[1799,:,32:])
plt.plot(y,x,"red",label = "Raft")
#plt.plot(yu,xu,"red")
plt.plot(y1,x1,"blue", label = "Transformer")
plt.plot(y2,x2,"black", label = "LSTM")
#plt.plot(y12,x12,"blue")
#plt.plot(y22,x22,"black")
plt.title("Blob Detection")
plt.legend()
plt.show()

plt.imshow(np.flip(label[196],axis=0))
plt.show()



lab_test1 = skimage.measure.find_contours(np.flip(orgdata50[1482],axis=0))
lab_test2 = skimage.measure.find_contours(np.flip(lstm_pred[1799],axis=0))
len(lab_test1)


x1 = lab_test1[0][:,0]
y1 = lab_test1[0][:,1]
x2 = lab_test2[0][:,0]
y2 = lab_test2[0][:,1]

x12 = lab_test1[1][:,0]
y12 = lab_test1[1][:,1]
x22 = lab_test2[1][:,0]
y22 = lab_test2[1][:,1]

x12

lstm_pred.shape

plt.imshow(np.flip(label2[1482], axis = 0))
plt.show()

plt.imshow(np.flip(orgdata50[1482],axis=0))
plt.show()

plt.imshow(np.flip(expx1[218], axis = 0))
plt.show()

import math

lifetime = []
label11 = label1.copy()
com = []
for i in range(len(np.unique(label1))-1):
    lifetime.append(len(np.unique(np.where(label1==i+1)[0])))
    com1 = []
    for j in np.unique(np.where(label1==i+1)[0]):
        label11[j][np.where(label1[j]==i+1)] = rawdata[j][np.where(label1[j]==i+1)]
        u = scipy.ndimage.center_of_mass(label11[j])
        if math.isnan(u[0]):
            None
        else:
            com1.append(u)
    com.append(com1)

xmap = np.zeros([64,64,2])
for i in range(63):
    xmap[i,i]-xmap[i+1,i+1]

for i in range(64):
    for j in range(64):
        xmap[i,j,0] = raw["r_arr"][i,j]
        xmap[i,j,1] = raw["z_arr"][i,j]

com_t = []

for i in range(len(com)):
    com_transformed1 = []
    for j in range(len(com[i])):
        y = int(com[i][j][0].round())
        x = int(com[i][j][1].round()) + 32
        com_transformed1.append(xmap[x,y])
    com_t.append(com_transformed1)

y_velocity = []
x_velocity = []

for i in range(len(com_t)):
    x_vel = 0
    y_vel = 0
    if len(com_t[i])>1:
        for j in range(len(com_t[i])-1):
            y_vel += com_t[i][j+1][0]-com_t[i][j][0]
            x_vel += com_t[i][j+1][1]-com_t[i][j][1]
    x_velocity.append(x_vel/len(com_t[i])/2.56e-06)
    y_velocity.append(y_vel/len(com_t[i])/2.56e-06)

plt.hist(y_velocity)
plt.title("Distribution of y-velocities")
plt.xlabel("Center of mass y-velocity/m/s")
plt.ylabel("Number of blobs")
plt.show()
plt.hist(x_velocity)
plt.title("Distribution of x-velocities")
plt.xlabel("Center of mass x-velocity/m/s")
plt.ylabel("number of blobs")
plt.show()

np.sort(x_velocity)

len(x_velocity)

np.mean((y_velocity))

abslo = []

for i in range(len(com)-1):
    abslo.append(label11[np.where(label1 == i+1)].max())

plt.hist(abslo)
plt.title("Distributions of maximum amplitude")
plt.xlabel("Amplitude")
plt.ylabel("Number of blobs")
plt.show()

countt = []

for i in range(np.max(label1)-1):
    if np.where(label1 == i+1)[0].size > 0:
        count = []
        for j in np.unique(np.where(label1==i+1)[0]):
            count.append(np.count_nonzero(label1[j] == i+1))
        countt.append(np.max(count))

np.mean(np.array(countt)*0.00102216*0.00084286)

plt.hist(np.array(countt)*0.00102216*0.00084286*10000)
#plt.plot(-np.sort(-np.random.exponential(0.62662514,10000)))
plt.title("Distributions of maximum size")
plt.xlabel("Maximum size/cm^2")
plt.ylabel("Number of blobs")
plt.show()

np.arange(len(np.array(countt))).shape
np.random.exponential([0.00564033, 0.37563047])

plt.plot(-np.sort(-np.random.exponential(0.62662514,10000)))
plt.show()

np.arange(len(np.array(countt)))

np.polyfit(np.arange(len(np.array(countt))),np.array(countt)*0.00102216*0.00084286*10000,0)



np.max(poly)

plt.imshow(np.flip(label11[1644], axis = 0))
plt.show()

np.max(rawdata)

sizes = np.zeros([2048,64,32, len(com)])

sizes[:,:,:,0][np.where(label1 == 1)] = 1

con

(raw["r_arr"][0].max()-raw["r_arr"][0].min())/64

raw["z_arr"]

raw["z_arr"][0]

plt.contourf(raw["r_arr"],raw["z_arr"], raw["brt_arr"][:,:,0])
plt.show()


