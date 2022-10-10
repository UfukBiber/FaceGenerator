import tensorflow as tf 
import os 

IMAGES_DIR = r"/home/ufuk/Desktop/img_align_celeba"

IMAGE_SIZE = (128, 128)



def ReadPaths(dir):
    imgPaths = os.listdir(dir)
    for i in range(len(imgPaths)):
        imgPaths[i] = os.path.join(dir, imgPaths[i])
    return imgPaths

def ReadImgsFromPaths(imgPath):
    img = tf.io.read_file(imgPath)
    img = tf.io.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    img = img / 255.
    return img 

def GetTrainDs():
    imagePaths = ReadPaths(IMAGES_DIR)[:30000]
    train_ds = tf.data.Dataset.from_tensor_slices((imagePaths), name = "train_ds")
    train_ds = train_ds.map(lambda path:ReadImgsFromPaths(path), num_parallel_calls=4)
    return train_ds 

