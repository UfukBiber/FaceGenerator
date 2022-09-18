import tensorflow as tf 
import os
import cv2

IMAGES_DIR = "IMAGES"
SAVE_DIR = r"C:\Users\biber\OneDrive\Desktop\FaceGenerator\OUTPUT_IMAGES"
CHECKPOINT_DIR = './ModelCheckpoint'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")
BATCH_SIZE = 4
BUFFER_SIZE = 4
BINARY_CROSS_ENTROPY = tf.keras.losses.BinaryCrossentropy(from_logits=True)

EPOCHS = 150
NOISE_DIM = 100
NOISE = tf.random.normal(shape = (BATCH_SIZE, NOISE_DIM))
NUM_EXAMPLE_TO_GENERATE = 1
SEED = tf.random.normal(shape = (NUM_EXAMPLE_TO_GENERATE, NOISE_DIM))

def ReadPaths(baseDir):
    paths = os.listdir(baseDir)
    for i in range(len(paths)):
        paths[i] = os.path.join(baseDir, paths[i])
    return paths

def PathToTensor(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels = 1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img 


def GetDataset(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda path : PathToTensor(path))
    ds = ds.batch(BATCH_SIZE).prefetch(BUFFER_SIZE)
    return ds 


def DiscriminatorLoss(realOutput, fakeOutput):
    realLoss = BINARY_CROSS_ENTROPY(tf.ones_like(realOutput), realOutput)
    fakeLoss = BINARY_CROSS_ENTROPY(tf.zeros_like(fakeOutput), fakeOutput)
    totalLoss = realLoss + fakeLoss
    return totalLoss

def GeneratorLoss(fakeOutput):
    return BINARY_CROSS_ENTROPY(tf.ones_like(fakeOutput), fakeOutput)


class Model:
    def __init__(self):
        self.generator = self.GenerativeModel()
        self.discriminator = self.DiscriminatorModel()
        self.generatorOptimizer = tf.keras.optimizers.Adam()
        self.discriminatorOptimizer = tf.keras.optimizers.Adam()
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generatorOptimizer,
                                 discriminator_optimizer=self.discriminatorOptimizer,
                                 generator=self.generator,
                                 discriminator=self.discriminator)

    
    
    def GenerativeModel(self): 
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(8*8*256, use_bias=False, input_shape=(NOISE_DIM,)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Reshape((8, 8, 256)))
        assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

        model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 8, 8, 128)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 64)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 32)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 64, 64, 1)

        return model

    
    def DiscriminatorModel(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[64, 64, 1]))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))
        return model

    @tf.function
    def trainStep(self, images):
        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
            NOISE = tf.random.normal(shape = (BATCH_SIZE, NOISE_DIM))
            generated_images = self.generator(NOISE, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = GeneratorLoss(fake_output)
            disc_loss = DiscriminatorLoss(real_output, fake_output)
            gradients_of_generator = genTape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = discTape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generatorOptimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminatorOptimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    
    def train(self, dataset):
        for epoch in range(EPOCHS):
            i = 0
            for imageBatch in dataset:
                self.trainStep(imageBatch)
                print ("Epoch %i / %i  Batch %i / 7500" % (epoch, EPOCHS, i), end = "\r")
                i += 1      
            print("")
            if (epoch % 1 == 0):
                self.SaveModel()
                print("Model is saved")

    def GenerateAndSaveImage(self):
        SEED = tf.random.normal(shape = (NUM_EXAMPLE_TO_GENERATE, NOISE_DIM))
        img = self.generator(SEED, training = False)
        img = img[0]
        img = tf.image.convert_image_dtype(img, tf.uint8)
        img = img.numpy() * 255
        isRunning = True 
        while isRunning :
            cv2.imshow("Image", img)
            key = cv2.waitKey(0)
            if (key == ord("q")):
                isRunning = False
            elif (key == ord("n")):
                SEED = tf.random.normal(shape = (NUM_EXAMPLE_TO_GENERATE, NOISE_DIM))
                img = self.generator(SEED, training = False)
                img = img[0]
                img = tf.image.convert_image_dtype(img, tf.uint8)
                img = img.numpy() * 255
        cv2.destroyAllWindows()
    
    def LoadModel(self):
        self.generator.load_weights("ModelCheckpoint\Generator")
        self.discriminator.load_weights("ModelCheckpoint\Discriminator")
    
    def SaveModel(self):
        self.generator.save_weights("ModelCheckpoint\Generator")
        self.discriminator.save_weights("ModelCheckpoint\Discriminator")
    


if __name__ == "__main__":
    imgPaths = ReadPaths(IMAGES_DIR)
    train_ds = GetDataset(imgPaths)
    model = Model()
    model.LoadModel()
    model.train(train_ds)
    
    


