import tensorflow as tf
import os 

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
        self.discriminator.trainable = False 
        self.gan = tf.keras.Sequential([
            self.generator,
            self.discriminator
        ])

        self.latent_dim = latentDim
    def compile(self):
        super(DCGAN, self).compile()
        self.disc_opt = tf.keras.optimizers.Adam(1e-4)
        self.gen_opt = tf.keras.optimizers.Adam(1e-4)
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.disc_loss_metric = tf.keras.metrics.Mean(name = "disc_met")
        self.gen_loss_metric = tf.keras.metrics.Mean(name = "gen_met")
        self.disc_acc_metrics = tf.keras.metrics.Accuracy(name = "disc_accuracy")
    
    @property
    def metrics(self):
        return [self.disc_loss_metric, self.gen_loss_metric]

    @tf.function
    def train_step(self, real_img_batch):
        batch_size = tf.shape(real_img_batch)[0]
        real_label = tf.ones(shape = (batch_size, 1), dtype = tf.float32)
        fake_label = tf.zeros(shape = (batch_size, 1), dtype = tf.float32)
        labels = tf.concat([real_label, fake_label], axis = 0)
        latent_vector = tf.random.normal(shape = (batch_size, self.latent_dim))
        gen_img_batch = self.generator(latent_vector)
        imgs = tf.concat([real_img_batch, gen_img_batch], axis = 0)
        with tf.GradientTape() as tape:
            predictions = self.discriminator(imgs)
            disc_loss = self.loss(labels, predictions)
        gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_opt.apply_gradients(zip(gradients, self.discriminator.trainable_variables))


        latent_vector = tf.random.normal(shape = (batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            gen_img_batch = self.generator(latent_vector)
            predictions = self.discriminator(gen_img_batch)
            gen_loss = self.loss(real_label, predictions)
        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(gradients, self.generator.trainable_variables))
        self.disc_loss_metric.update_state(disc_loss)
        self.gen_loss_metric.update_state(gen_loss)
        return {"disc_met":self.disc_loss_metric.result(), "gen_met":self.gen_loss_metric.result()}

    def SaveModel(self):
        self.generator.save_weights(os.path.join("DCGAN", "MODEL"))
        self.discriminator.save_weights(os.path.join("DCGAN", "MODEL"))


class CallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
    
    def on_epoch_end(self, epoch, logs=None):
        latent_vectors = tf.random.normal(shape=(3, self.model.latent_dim))
        generated_images = self.model.generator(latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(3):
            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
            img.save(os.path.join("GeneratedImages", "generated_img_%03d_%d.png" % (epoch, i)))
        
        self.model.SaveModel()