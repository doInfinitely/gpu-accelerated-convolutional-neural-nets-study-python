import time
import math
import numpy as np
from numba import cuda
from multiprocessing import Process
from preprocess import constructTensors
import pickle
import random
from multiprocessing import Process
import os

MAXTHREADSPERBLOCK = 1024

@cuda.jit
def forward_convolutional(Weights, tile, Input, Output):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = (tx+ty*bw) % tile[0]
    j = (tx+ty*bw) // tile[0] % tile[1]
    k = (tx++ty*bw) // tile[0] // tile[1] % (Input.shape[0]-tile[0])
    l = (tx++ty*bw) // tile[0] // tile[1] // (Input.shape[0]-tile[0]) % (Input.shape[1]-tile[1])
    m = (tx++ty*bw) // tile[0] // tile[1] // (Input.shape[0]-tile[0]) // (Input.shape[1]-tile[1])
    if m < Input.shape[2]:
        cuda.atomic.add(Output, (k,l,m), Input[k+i,l+j,m]*Weights[i,j])

@cuda.jit
def backward_convolutional(Weights, tile, Derivative, Input, dError, dInput):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = (tx+ty*bw) % tile[0]
    j = (tx+ty*bw) // tile[0] % tile[1]
    k = (tx++ty*bw) // tile[0] // tile[1] % (Input.shape[0]-tile[0])
    l = (tx++ty*bw) // tile[0] // tile[1] // (Input.shape[0]-tile[0]) % (Input.shape[1]-tile[1])
    m = (tx++ty*bw) // tile[0] // tile[1] // (Input.shape[0]-tile[0]) // (Input.shape[1]-tile[1])
    if m < Input.shape[2]:
        cuda.atomic.add(Derivative, (i,j), Input[k+i,l+j,m]*dError[k,l,m])
        cuda.atomic.add(dInput, (k+i,l+j,m), dError[k,l,m]*Weights[i,j])

class ConvolutionalLayer():
    def __init__(self, tile):
        self.weights = np.array([[np.random.normal() for x in range(tile[1])] for y in range(tile[0])])
        self.tile = np.array(tile)
        self.derivative = np.zeros(tile)
    def forward(self, Input):
        tile = self.tile
        threadsperblock = min(MAXTHREADSPERBLOCK, tile[0]*tile[1]*Input.shape[0]*Input.shape[1])
        blockspergrid = math.ceil(tile[0]*tile[1]*Input.shape[0]*Input.shape[1]*Input.shape[2]/threadsperblock)
        self.input = Input
        self.output = np.zeros((Input.shape[0]-tile[0], Input.shape[1]-tile[1], Input.shape[2]))
        forward_convolutional[blockspergrid, threadsperblock](self.weights, self.tile, Input, self.output)
        return self.output
    def backward(self, dError):
        tile = self.tile
        threadsperblock = min(MAXTHREADSPERBLOCK, tile[0]*tile[1]*Input.shape[0]*Input.shape[1])
        blockspergrid = Input.shape[2]
        self.input = Input
        dInput = np.zeros((self.input.shape[0], self.input.shape[1], self.input.shape[2]))
        backward_convolutional[blockspergrid, threadsperblock](self.weights, self.tile, self.derivative, self.input, dError, dInput)
        return dInput
    def gradient(self, delta):
        self.weights -= delta*self.derivative
        self.derivative = np.zeros(self.tile)
    def dump(self, filename):
        pickle.dump((self.weights, self.tile), open(filename, 'wb'))
    def load(self, filename):
        self.weights, self.tile = pickle.load(open(filename, 'rb'))

@cuda.jit
def forward_sigmoidal(Input, Output):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = (tx+ty*bw) % Input.shape[0]
    j = (tx+ty*bw) // Input.shape[0] % Input.shape[1]
    k = (tx+ty*bw) // Input.shape[0] // Input.shape[1]
    if k < Input.shape[2]:
        Output[i,j,k] = 1.0/(1+math.exp(float(-1*Input[i,j,k])))

@cuda.jit
def backward_sigmoidal(dError, Output, dInput):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = (tx+ty*bw) % dError.shape[0]
    j = (tx+ty*bw) // dError.shape[0] % dError.shape[1]
    k = (tx+ty*bw) // dError.shape[0] // dError.shape[2]
    if k < dError.shape[2]:
        dInput[i,j,k] = Output[i,j,k]*(1-Output[i,j,k])*dError[i,j,k]

class Sigmoidal():
    def __init__(self):
        pass
    def forward(self, Input):
        self.input = Input
        try:
            if len(Input.shape) == 1:
              raise AttributeError
            threadsperblock = min(MAXTHREADSPERBLOCK, Input.shape[0]*Input.shape[1])
            blockspergrid = Input.shape[2]
            self.output = np.zeros((Input.shape[0], Input.shape[1], Input.shape[2]))
            forward_sigmoidal[blockspergrid, threadsperblock](Input, self.output)
            return self.output
        except AttributeError:
            self.output = np.divide(np.ones(self.dim), np.add(np.exp(np.multiply(self.input,-1)),1))
            return self.output
    def backward(self, dError):
        try:
            if len(dError.shape) == 1:
                raise AttributeError
            threadsperblock = min(MAXTHREADSPERBLOCK, self.input.shape[0]*self.input.shape[1])
            blockspergrid = self.input.shape[2]
            dInput = np.zeros((dError.shape[0], dError.shape[1]))
            backward_sigmoidal[blockspergrid, threadsperblock](dError, self.output, dInput)
            return dInput
        except AttributeError:
            return np.multiply(np.multiply(self.output, np.subtract(1, self.output)), vec)
    def gradient(self, delta):
        pass
    def revert(self):
        pass

@cuda.jit
def forward_sigmoidal_cuda(Input, Output):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = (tx+ty*bw) % Input.shape[0]
    j = (tx+ty*bw) // Input.shape[0]
    if j < Input.shape[1]:
        Output[i,j] = 1.0/(1+math.exp(float(-1*Input[i,j])))

@cuda.jit
def backward_sigmoidal_cuda(vec, Output, dInput):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = (tx+ty*bw) % vec.shape[0]
    j = (tx+ty*bw) // vec.shape[0]
    if j < vec.shape[1]:
        dInput[i,j] = Output[i,j]*(1-Output[i,j])*vec[i,j]

class Sigmoidal():
    def __init__(self):
        pass

    def forward(self, Input):
        self.input = Input
        try:
            if len(Input.shape) == 1:
                raise AttributeError
            threadsperblock = min(MAXTHREADSPERBLOCK, Input.shape[0])
            blockspergrid = math.ceil(Input.shape[0]*Input.shape[1]/threadsperblock)
            self.output = np.zeros((Input.shape[0], Input.shape[1]))
            forward_sigmoidal_cuda[blockspergrid, threadsperblock](Input, self.output)
            return self.output
        except AttributeError:
            self.output = np.divide(np.ones(self.dim), np.add(np.exp(np.multiply(self.input,-1)),1))
            return self.output

    def backward(self, vec):
        try:
            if len(vec.shape) == 1:
                raise AttributeError
            threadsperblock = min(MAXTHREADSPERBLOCK, self.input.shape[1])
            blockspergrid = math.ceil(self.input.shape[0]*self.input.shape[1]**2/threadsperblock)
            dInput = np.zeros((vec.shape[0], vec.shape[1]))
            backward_sigmoidal_cuda[blockspergrid, threadsperblock](vec, self.output, dInput)
            return dInput
        except AttributeError:
            return np.multiply(np.multiply(self.output, np.subtract(1, self.output)), vec)

    def gradient(self, delta):
        pass

    def revert(self):
        pass

@cuda.jit
def forward_layer_cuda(Weights, Input, Output):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = (tx+ty*bw) % Input.shape[0]
    j = (tx+ty*bw) // Input.shape[0] % Weights.shape[1]
    k = (tx+ty*bw) // Input.shape[0] // Weights.shape[1]
    if k < Input.shape[1]:
        cuda.atomic.add(Output, (j,k), Input[i,k]*Weights[i,j])

@cuda.jit
def backward_layer_cuda(Weights, Derivative, Input, dError, dInput):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = (tx+ty*bw) % Input.shape[0]
    j = (tx+ty*bw) // Input.shape[0] % Weights.shape[1]
    k = (tx+ty*bw) // Input.shape[0] // Weights.shape[1]
    if k < Input.shape[1]:
        cuda.atomic.add(Derivative, (i,j), Input[i,k]*dError[j,k])
        cuda.atomic.add(dInput, (i,k), dError[j,k]*Weights[i,j])

class Layer():
    def __init__(self, inputDim, outputDim):
        self.dim = [inputDim, outputDim]
        self.weights = np.multiply(np.random.normal(size=(self.dim[0], self.dim[1])), 1)
        self.derivative = np.zeros((self.dim[0], self.dim[1]))
        self.input = np.zeros(self.dim[0])
        self.output = np.zeros(self.dim[1])

    def forward(self, Input):
        self.input = Input
        try:
            if len(Input.shape) == 1:
                raise AttributeError
            threadsperblock = min(MAXTHREADSPERBLOCK, self.dim[0])
            blockspergrid = math.ceil(self.dim[0]*self.dim[1]*Input.shape[1]/threadsperblock)
            self.output = np.zeros((self.dim[1], Input.shape[1]))
            forward_layer_cuda[blockspergrid, threadsperblock](self.weights, Input, self.output)
        except AttributeError:
            self.output = np.matmul(self.input, self.weights)
        return self.output

    def backward(self, dError):
        try:
            if len(self.input.shape) == 1:
                raise AttributeError
            threadsperblock = min(MAXTHREADSPERBLOCK, self.input.shape[0])
            blockspergrid = math.ceil(self.input.shape[0]*self.input.shape[1]**2/threadsperblock)
            dInput = np.zeros((self.dim[0], self.input.shape[1]))
            backward_layer_cuda[blockspergrid, threadsperblock](self.weights, self.derivative, self.input, dError, dInput)
            return dInput
        except AttributeError:
            for i in range(self.dim[0]+1):
                for j in range(self.dim[1]):
                    self.derivative[i,j] += self.input[i]*dError[j]
            return np.multiply(dError, self.weights)

    def gradient(self, delta):
        #self.prevW = copy.deepcopy(self.weights)
        self.weights = np.add(self.weights, np.multiply(-1*delta, self.derivative))
        self.derivative = np.zeros((self.dim[0], self.dim[1]))

    def revert(self):
        self.weights = self.prevW
    def dump(self, filename):
        pickle.dump((self.dim, self.weights), open(filename, 'wb'))
    def load(self, filename):
        self.dim, self.weights = pickle.load(open(filename, 'rb'))

class Loss():
    def __init__(self, target):
        self.target = target
        self.dError = None
    def forward(self, Input):
        self.input = Input
        output = 0
        for i,x in enumerate(self.target):
            for j, y in enumerate(self.target[i]):
                if Input[i,j] != 0:
                    output -= y*math.log(Input[i,j], 2)
                if Input[i,j] != 1:
                    output -= (1-y)*math.log(1-Input[i,j],2)
        self.output = output/len(self.target)
        return self.output

    def backward(self):
        if self.dError == None:
            self.dError = np.zeros(self.input.shape)
        for i,x in enumerate(self.target):
            for j,y in enumerate(self.target[i]):
                if self.input[i,j] != 0:
                    self.dError[i,j] += y/(self.input[i,j]*math.log(2))
                if self.input[i,j] != 1:
                    self.dError[i,j] += (1-y)/((1-self.input[i,j])*math.log(2))
        return self.dError

    def gradient(self, delta=0):
        self.dError = None

class Model():
    def __init__(self, target=None):
        self.convs = [ConvolutionalLayer((5,5)) for x in range(10)]
        self.sigmo = Sigmoidal()
        self.dense = Layer(27*27*len(self.convs)+1,10)
        self.sigmo2 = Sigmoidal()
        self.loss = Loss(target)
    def forward(self, Input, target=None):
        if not target is None:
            self.loss = Loss(target)
        temp = None
        for x in self.convs:
            temp2 = x.forward(Input)
            try:
                temp = np.concatenate((temp, temp2), axis=1)
            except ValueError:
                temp = temp2
        temp = temp.reshape((temp.shape[0]*temp.shape[1], temp.shape[2]))
        temp = self.sigmo.forward(temp)
        temp = np.concatenate((temp, np.ones((1,temp.shape[1]))), axis=0)
        temp = self.dense.forward(temp)
        temp = self.sigmo2.forward(temp)
        if not self.loss.target is None:
            print(self.loss.forward(temp))
        return temp
    def backward(self):
        temp = self.loss.backward()
        temp = self.sigmo2.backward(temp)
        temp = self.dense.backward(temp)
        temp = temp[:temp.shape[0]-1, :]
        np.ascontiguousarray(temp)
        temp = self.sigmo.backward(temp)
        for i,x in enumerate(self.convs):
            #print(temp[i*27*27:(i+1)*27*27,:].shape)
            temp2 = temp[i*27*27:(i+1)*27*27,:].reshape(27, 27, temp.shape[1])
            np.ascontiguousarray(temp2)
            x.backward(temp2)
    def gradient(self, delta):
        self.loss.gradient(delta)
        self.sigmo2.gradient(delta)
        self.dense.gradient(0)
        self.sigmo.gradient(delta)
        for i,x in enumerate(self.convs):
            x.gradient(delta)
    def dump(self, foldername):
        if not os.path.isdir(foldername):
            os.mkdir(foldername)
        filename = 'parameters.p'
        splitfile = filename.split('.')
        fileTemplate = splitfile[-2]+'_{0}.'+splitfile[-1]
        for i,x in enumerate(self.convs):
            x.dump(os.path.join(foldername,fileTemplate.format('conv_{0}'.format(i))))
        self.dense.dump(os.path.join(foldername,fileTemplate.format('dense')))
    def load(self, foldername):
        filename = 'parameters.p'
        splitfile = filename.split('.')
        fileTemplate = splitfile[-2]+'_{0}.'+splitfile[-1]
        for i,x in enumerate(self.convs):
            x.load(os.path.join(foldername, fileTemplate.format('conv_{0}'.format(i))))
        self.dense.load(os.path.join(foldername,fileTemplate.format('dense')))

if __name__ == '__main__':
    Input, Output = constructTensors()
    model = Model()
    start = time.time()
    count = 0
    while True:
        losses = []
        for i in range(500):
            index = 0#random.randrange(Output.shape[1]-100)
            model.forward(np.ascontiguousarray(Input[:,:,index:index+100]), np.ascontiguousarray(Output[:,index:index+100]))
            losses.append(model.loss.output)
            model.backward()
            model.gradient(0.00000001)
        print(model.sigmo2.output)
        if losses[0] < losses[-1]:
            break
        model.dump('parameters')
