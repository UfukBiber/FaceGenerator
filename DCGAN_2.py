import tensorflow as tf

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


generator =  tf.keras.Sequential(
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

class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self):
        super(GAN, self).compile()
        self.d_optimizer = tf.keras.optimizers.Adam(1e-5)
        self.g_optimizer = tf.keras.optimizers.Adam(1e-5)
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
        self.generator.save_weights("DCGAN_2\Generator")
        self.discriminator.save_weights("DCGAN_2\Discriminator")

    def LoadModel(self):
        try:
            self.generator.load_weights("DCGAN_2\Generator")
            self.discriminator.load_weights("DCGAN_2\Discriminator")
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
            img.save("generated_img_%03d_%d.png" % (epoch, i))
        self.model.SaveModel()

    
