from PIL import Image, ImageDraw

img = Image.new('L', (28, 28), 0)
pixels = img.load()

with open('mnist_train.csv') as file:
    for line in file:
        label = None
        for i,x in enumerate(line.split(',')):
            if i == 0:
                label = x
            else:
                pixels[(i-1)%28,(i-1)//28] = int(x)
        print(label)
        img.show()
        break
