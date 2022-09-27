import tensorflow as tf 
import os 

# def PreprocessImages(oldDir, newDir, imgSize):
#     if not (os.path.exists(newDir)):
#         os.mkdir(newDir)
#     print("PreprocessingImages")
#     imgPaths = os.listdir(oldDir)
#     length = len(imgPaths)
#     i = 1
#     for imgPath in imgPaths:
#         oldImgPath = os.path.join(oldDir, imgPath)
#         newImgPath = os.path.join(newDir, imgPath)
#         img = tf.io.read_file(oldImgPath)
#         img = tf.io.decode_jpeg(img, 3)
#         img = tf.image.resize(img, (64, 64))
#         tf.keras.utils.save_img(newImgPath, img)
#         print("%i / %i is processed"%(i, length), end = "\r")
#         i += 1
#     print("Done")



def ReadPathDataset(dir):
    print("Reading image paths")
    imgPaths = os.listdir(dir)[0:100000]
    for i in range(len(imgPaths)):
        imgPaths[i] = os.path.join(dir, imgPaths[i])
    ds = tf.data.Dataset.from_tensor_slices(imgPaths)
    print("Done")
    return ds

def ReadImgsFromPaths(imgPath, imgSize, channels = 3):
    img = tf.io.read_file(imgPath)
    img = tf.io.decode_jpeg(img, channels)
    img = tf.image.resize(img, imgSize)
    img = tf.image.convert_image_dtype(img, tf.float32) / 255.
    return img 


