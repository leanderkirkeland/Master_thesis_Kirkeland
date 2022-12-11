import tensorflow as tf
import numpy as np 
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from Code.LSTM.utils import accuracy, IoU2

def LSTM32x64(input_size = (100,64,32,1)):
  input = Input(input_size) 
  LSTM1 = ConvLSTM2D(8, 3, return_sequences=True, padding="same",activation="relu")(input)
  Pool1 = TimeDistributed(MaxPooling2D((2,2), strides=(2,2)))(LSTM1)

  LSTM4 = ConvLSTM2D(16, 3, return_sequences=True, padding="same",activation="relu")(Pool1)

  Pool4 = TimeDistributed(UpSampling2D((2,2)))(LSTM4)
  CONV4 = Conv2D(16, 3 , padding = "same", activation="relu")(Pool4)
  output = Conv2D(10, 3 , padding = "same", activation = "softmax")(CONV4)

  model = Model(inputs = input, outputs = [output])
  model.compile(loss='sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(), metrics=["accuracy"], sample_weight_mode='temporal')
  return model


"""
class LSTM():
  def __init__():
    None
  
  def preprocess_data(self, dir = "Dataset", ns = 3000, vs = 500):
    train_x = []
    train_y = []
    val_x = []
    val_y = []

      
    for i in range(ns):
        data = xr.open_dataset(f"{dir}/32x64test{i}.nc")
        train_x.append(data["n"])
        train_y.append(data["blob_labels"])

    for i in range(vs):
        data = xr.open_dataset(f"{dir}/32x64test{i+ns}.nc").to_array()
        val_x.append(data[0])
        val_y.append(data[1])

    train_x = np.moveaxis(np.array(train_x).reshape(ns,64,32,100,1),3,1) #makes it shape (n, 100, 64, 32, 1)
    train_y = np.moveaxis(np.array(train_y).reshape(ns,64,32,100,1),3,1)
    val_x = np.moveaxis(np.array(val_x).reshape(vs,64,32,100,1),3,1)
    val_y = np.moveaxis(np.array(val_y).reshape(vs,64,32,100,1),3,1)

    train_x = train_x/np.max(train_x)
    val_x = val_x/np.max(val_x)  

    return train_x, val_x, train_y, val_y

  def get_test_data(self):
    test_x = []
    test_y = []
    for i in range(500):
      data = xr.open_dataset(f"{dir}/32x64test{i+5700}.nc").to_array()
      test_x.append(data[0])
      test_y.append(data[1])

    test_x = np.moveaxis(np.array(test_x).reshape(500,64,32,100,1),3,1)
    test_y = np.moveaxis(np.array(test_y).reshape(500,64,32,100,1),3,1)

    test_x = test_x/np.max(test_x)

    return test_x, test_y

  def train(self, dir = "Dataset", ns = 3000, pretrained = False, save = False, name = "LSTMtest", epochs = 20, batchsize = 8):
    if pretrained:
      self.model = load_model("LSTMtest")
    else:
      self.model = LSTM32x64()

    train_x, val_x, train_y, val_y = self.preprocess_data()
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    weights = np.where(train_y == 0, 0.1, 0.9)
    self.model.fit(train_x, train_y, epochs=epochs, batch_size=batchsize, sample_weight=weights, validation_data=(val_x, val_y), callbacks=[tensorboard_callback])

    if save = True:
      self.model.save(name)

    return self.model
  def load_model(self, location = "LSTMtest"):
    self.model = load_model(location)
    return self.model

  def predict(self, data = None, experimental = False):
    if experimental == True:
      
    if data == None:
      test_x, test_y = self.get_test_data()
    else:
      test_x, test_y = data
    
    x = self.model.predict(test_x)
    shape = x.shape[0]*x.shape[1]
    iouC = IoU2(test_y.reshape(shape,64,32), x.reshape(shape,64,32),16)
    acc = accuracy(test_y.reshape(shape,64,32), x.reshape(shape,64,32),16)
    metric = tf.keras.metrics.MeanIoU(num_classes=11)
    IoU = metric(test_y.reshape(shape,64,32), x.reshape(shape,64,32))

    return iouC, acc, IoU


p = np.zeros(test_x.shape).shape[0]
p/100
"""