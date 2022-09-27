import tensorflow as tf
import numpy as np
import Input

LATENT_DIM = 128

class DCGAN(tf.keras.models.Model):
    def __init__(self, latentDim = 128):
        super(DCGAN, self).__init__()
        self.discriminator = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(64, 64, 3)),
                tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ],
            name="discriminator",
        )
        self.generator = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(128,)),
                tf.keras.layers.Dense(8 * 8 * 128),
                tf.keras.layers.Reshape((8, 8, 128)),
                tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
            ],
            name="generator",
        )
        self.latentDim = latentDim
        
    def compile(self):
        super(DCGAN, self).compile()
        self.d_optimizer = tf.keras.optimizers.Adam()
        self.g_optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.discLossMet = tf.keras.metrics.Mean(name='disc_loss')
        self.genLossMet = tf.keras.metrics.Mean(name='gen_loss')

    @tf.function
    def train_step(self, realImgBatch):
        batchSize = tf.shape(realImgBatch)[0]
        realLabel = tf.ones((batchSize, 1))
        fakeLabel = tf.zeros((batchSize, 1))
        latentVector = tf.random.normal((batchSize, self.latentDim))
        genImgBatch = self.generator(latentVector)
        with tf.GradientTape() as discTape:
            realOutput = self.discriminator(realImgBatch)
            fakeOutput = self.discriminator(genImgBatch)
            realLoss = self.loss_fn(realLabel, realOutput)
            fakeLoss = self.loss_fn(fakeLabel, fakeOutput)
            discLoss = (realLoss + fakeLoss) * 0.5
        gradients = discTape.gradient(discLoss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        latentVector = tf.random.normal((batchSize, self.latentDim))
        with tf.GradientTape() as genTape:
            genImg = self.generator(latentVector)
            fakeOut = self.discriminator(genImg)
            genLoss = self.loss_fn(realLabel, fakeOut)
        gradients = genTape.gradient(genLoss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))


        self.discLossMet(discLoss)
        self.genLossMet(genLoss)

    def train(self, ds, epochs):
        self.discLossMet.reset_states()
        for epoch in range(epochs):
            i = 0
            for realImgBatch in ds:
                self.train_step(realImgBatch)
                i+=1
                print("%i / %i  %i / %i  %2f  %2f"%(epoch, epochs,i, 6000, self.discLossMet.result(), self.genLossMet.result()), end = "\r")

    def SaveModel(self):
        self.generator.save_weights("DCGAN\Generator")
        self.discriminator.save_weights("DCGAN\Discriminator")

    def LoadModel(self):
        try:
            self.generator.load_weights("DCGAN\Generator")
            self.discriminator.load_weights("DCGAN\Discriminator")
        except:
            pass


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("GeneratedImages\generated_img_%03d_%d.png" % (epoch, i))
        self.model.SaveModel()


