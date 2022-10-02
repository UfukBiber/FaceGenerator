import tensorflow as tf 
from DCGAN_Model2 import GAN
import matplotlib.pyplot as plt 


NUM_OF_EXAMPLES = 12
NOISE_DIM = 128



def PlotImages(model):
    noise = tf.random.normal(shape = (NUM_OF_EXAMPLES, NOISE_DIM))
    output = model.generator.predict(noise)
    output = output * 255.
    output = output.numpy().astype("uint8")
    fig, ax = plt.subplots(3, 4)
    for i in range(3):
        for j in range(4):
            ax[i, j].imshow(output[i*3 + j])
    plt.show()


if __name__ == "__main__":
    model = GAN()
    model.LoadModel()
    PlotImages(model)
    