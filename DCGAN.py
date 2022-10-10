import tensorflow as tf
import os 
from tensorflow.keras import layers

EPOCH = 0


class GAN(tf.keras.Model):
    def __init__(self):
        super(GAN, self).__init__()
        self.discriminator = self.GetDiscriminator()
        self.generator = self.GetGenerator()
        self.latent_dim = 128

    def GetGenerator(self):
        generator = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(128,)),
            layers.Dense(8 * 8 * 128),
            layers.Reshape((8, 8, 128)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
        ],
        name="generator")
        return generator
    
    def GetDiscriminator(self):
        discriminator = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(128, 128, 3)),
                layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="discriminator",
        )
        return discriminator

    def compile(self):
        super(GAN, self).compile()
        self.d_optimizer = tf.keras.optimizers.Adam(0.0001)
        self.g_optimizer = tf.keras.optimizers.Adam(0.0001)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


    def SaveModel(self):
        self.generator.save_weights(os.path.join("Generator", "GENERATOR"))
        self.discriminator.save_weights(os.path.join("Discriminator", "DISCRIMINATOR"))
    
    def LoadModel(self):
        try:
            self.generator.load_weights(os.path.join("Generator", "GENERATOR"))
            self.discriminator.load_weights(os.path.join("Discriminator", "DISCRIMINATOR"))
            print("Loaded the weights.")
        except:
            print("Using new model") 
            pass
def SaveEpochs(epoch):
    with open("Epochs.txt", "w") as f:
        f.write(str(epoch))
        f.close()
        
def LoadEpochs():
    global EPOCH
    with open("Epoch.txt", "r") as f:
        EPOCH = int(f.readline())
        f.close()

class CallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        if (os.path.exists("Epoch.txt")):
            LoadEpochs()
    
    def on_epoch_end(self, epoch, logs=None):
        latent_vectors = tf.random.normal(shape=(3, self.model.latent_dim))
        generated_images = self.model.generator(latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(3):
            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
            img.save(os.path.join("GeneratedImages", "generated_img_%03d_%d.png" % (epoch+EPOCH, i)))
        self.model.SaveModel()
        SaveEpochs(epoch+EPOCH+1)