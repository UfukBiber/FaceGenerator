from gc import callbacks
import tensorflow as tf 
import Input
import DCGAN_Model2




BATCH_SIZE = 32
BUFFER_SIZE = 16
EPOCHS = 20


if __name__ == "__main__":
    train_ds = Input.GetTrainDs()
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(BUFFER_SIZE)
    model = DCGAN_Model2.GAN()
    model.compile()
    model.LoadModel()
    model.fit(train_ds, epochs = 10, callbacks = [DCGAN_Model2.CallBack()])