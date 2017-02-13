import random
import math

# simple neuron with two inputs


def sigmoid(value):
    return 1 / (1 + math.e**-value)


def sigmoid_derivative(value):
    return sigmoid(value) * (1 - sigmoid(value))


trainingInputs = ((100, 30), (400, 300), (600, 10), (55, 500))
trainingOutputs = (0, 1, 0, 1)
learningRate = 1;
weight1 = random.randint(0, 100) / 100
weight2 = random.randint(0, 100) / 100

print(weight1)
print(weight2)

for i in range(0,1000):
    for dataSet in range(0, len(trainingOutputs)):
        input1 = trainingInputs[dataSet][0]
        input2 = trainingInputs[dataSet][1]
        desiredOutput = trainingOutputs[dataSet]
        summedInput = input1 * weight1 + input2 * weight2
        print("summedInput: " + str(summedInput))
        #sigmoid function
        output = sigmoid(summedInput)
        print("output: " + str(output))
        error = 0.5 * (desiredOutput - output)**2
        print("error: " + str(error))
        weight1Change = -learningRate * input1 * sigmoid_derivative(summedInput) * (output - desiredOutput)
        weight2Change = -learningRate * input2 * sigmoid_derivative(summedInput) * (output - desiredOutput)

        print("weight1Change: " + str(weight1Change))
        print("weight2Change: " + str(weight2Change))

        weight1 += weight1Change
        weight2 += weight2Change

        print("weight1: " + str(weight1))