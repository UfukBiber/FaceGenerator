import tensorflow as tf 
import Input
import DCGAN_Model
import DCGAN_2



DIR = r"C:\Users\biber\OneDrive\Desktop\IMAGES"
IMAGE_SIZE= (64, 64)
BATCH_SIZE = 32
BUFFER_SIZE = 16
EPOCHS = 20
if __name__ == "__main__":
    # Input.PreprocessImages(DIR, "IMAGES", IMAGE_SIZE)
    ds = Input.ReadPathDataset(DIR)
    ds = ds.map(lambda imgPath:Input.ReadImgsFromPaths(imgPath, IMAGE_SIZE, 3))
    ds = ds.batch(BATCH_SIZE).prefetch(BUFFER_SIZE)
    model = DCGAN_2.GAN(DCGAN_2.discriminator, DCGAN_2.generator, 128)
    model.compile()
    model.LoadModel()
    model.fit(
    ds, epochs=10, callbacks=[DCGAN_2.GANMonitor(num_img=3, latent_dim=128)]
)