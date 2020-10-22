from train import Model
from preprocess import constructTensors

model = Model()
model.load('parameters')
Input, Output = constructTensors()
print(model.forward(Input))
print(Output)
