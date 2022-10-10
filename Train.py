
import tensorflow as tf 
import Input
import DCGAN


BATCH_SIZE = 4
BUFFER_SIZE = 1
EPOCHS = 20








if __name__ == "__main__":
    train_ds = Input.GetTrainDs()
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(BUFFER_SIZE)
    model = DCGAN.GAN()
    model.compile()
    model.LoadModel()
    model.fit(train_ds, epochs = EPOCHS, callbacks = [DCGAN.CallBack()])