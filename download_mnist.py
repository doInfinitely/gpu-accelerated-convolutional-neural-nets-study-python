import subprocess

urls = dict()
urls['train-images'] = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
urls['train-labels'] = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
urls['10k-images'] = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
urls['10k-labels'] = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

for key in urls:
    subprocess.call(["wget", urls[key]])
    subprocess.call(["gunzip", urls[key].split('/')[-1]])
