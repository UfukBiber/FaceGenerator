import tensorflow as tf
import os 

LATENT_DIM = 128

class Discriminator():
    def __init__(self, *args, **kwargs):
        self.discriminator = self.GetDiscriminator()
        self.gan = self.GetGanModel()

    
    def GetDiscriminator(self):
        discriminator = tf.keras.Sequential(
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
        return discriminator
    
    def GetGanModel(self):
        self.discriminator.trainable = False 
        model = tf.keras.Sequential([
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
            self.discriminator
        ], name = "ganModel")
        return model 
    
    def compile(self):
        self.discriminator.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
        self.gan.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    