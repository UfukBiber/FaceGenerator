import cv2 
import os


DATA_DIR = r"C:\Users\biber\OneDrive\Desktop\CelebA-HQ-img"
IMAGE_DIR = "IMAGES"
SIZE = (64,  64)


def ResizeAndConvertGrayScale(img, SIZE):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, SIZE)
    return img 

def ReadImg(filePath):
    img = cv2.imread(filePath)
    return img 

def SaveImg(img, filePath):
    cv2.imwrite(filePath, img)







if __name__ == "__main__":
    print("\n\nResizing to (%i, %i), converting grayscale and saving to %s folder.\n\n" % (SIZE[0], SIZE[1], IMAGE_DIR))
    if not (os.path.exists(IMAGE_DIR)):
        os.mkdir(IMAGE_DIR)
    filePaths = os.listdir(DATA_DIR)
    i = 0
    lenFilePaths = len(filePaths)
    for filePath in filePaths:
        img = ReadImg(os.path.join(DATA_DIR, filePath))
        img = ResizeAndConvertGrayScale(img, SIZE)
        SaveImg(img, os.path.join(IMAGE_DIR, filePath))
        print("%i of %i images were processed" % (i, lenFilePaths), end = "\r")
        i+=1
