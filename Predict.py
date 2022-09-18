import tensorflow as tf 
from Model import Model



model = Model()


if __name__ == "__main__":
    model.LoadModel()
    model.GenerateAndSaveImage()