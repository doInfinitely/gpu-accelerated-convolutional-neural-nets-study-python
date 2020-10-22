import numpy as np
from PIL import Image

def constructTensors():
    Input = None
    Output = None
    with open('mnist_train.csv') as file:
        for line in file:
            label = None
            img = Image.new('L', (28, 28), 0)
            pixels = img.load()
            label = None
            for i,x in enumerate(line.split(',')):
                if i == 0:
                    label = int(x)
                else:
                    pixels[(i-1)%28,(i-1)//28] = int(x)
            array = np.array(img)/255
            array = np.reshape(array, (array.shape[0], array.shape[1], 1))
            try:
                Input = np.concatenate((Input, array), axis=2)
            except ValueError:
                Input = array
            array = np.array([1 if label == i else 0 for i in range(10)])
            array = np.reshape(array, (10,  1))
            try:
                Output = np.concatenate((Output, array), axis=1)
            except ValueError:
                Output = array
    return Input, Output

if __name__ == "__main__":
    Input, Output = constructTensors()
    print(Input.shape, Output.shape)
