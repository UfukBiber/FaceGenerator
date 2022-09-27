import tensorflow as tf 
from DCGAN_Model import DCGAN
import cv2


NUM_OF_EXAMPLES = 36
NOISE_DIM = 128
model = DCGAN()
model.LoadModel()


def PlotImages(model):
    noise = tf.random.normal(shape = (NUM_OF_EXAMPLES, NOISE_DIM))
    output = model.generator(noise, training = False)
    output = output.numpy().astype("uint8")
    isRunning = True 
    i = 0
    while isRunning:
        cv2.imshow("GAN", output[i])
        key = cv2.waitKey(0)
        if (key == ord("q")):
            isRunning = False 
        elif (key == ord("n")):
            i+=1 
        
    cv2.destroyAllWindows()


if __name__ == "__main__":
    PlotImages(model, )
    