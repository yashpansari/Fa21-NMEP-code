import numpy as np

class NeuralNet:

    #constructor, we will hardcode this to a 1 hidden layer network, for simplicity
    #the problem we will grade on is differentiating 0 and 1s
    #Some things/structuure may need to be changed. What needs to stay consistant is us being able to call
    #forward with 2 arguments: a data point and a label. Strange architecture, but should be good for learning
    def __init__(self, input_size=784, hidden_size=100, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        #YOUR CODE HERE, initialize appropriately sized weights/biases with random paramters
        self.weight1 = np.random.randn(self.input_size, self.hidden_size)
        self.bias1 = np.random.randn(self.hidden_size)
        self.weight2 = np.random.randn(self.hidden_size, self.output_size)
        self.bias2 = np.random.randn(self.output_size)

    #Potentially helpful, np.dot(a, b), also @ is the matrix product in numpy (a @ b)

    #loss function, implement L1 loss
    #YOUR CODE HERE
    def loss(self, y0, y1):
        return np.absolute(y0 - y1)
    #relu and sigmoid, nonlinear activations
    #YOUR CODE HERE
    def relu(self, x):
        x[x<0] = 0
        return x

    #You also may want the derivative of Relu and sigmoid

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    #forward function, you may assume x is correct input size
    #have the activation from the input to hidden layer be relu, and from hidden to output be sigmoid
    #have your forward function call backprop: we won't be doing batch training, so for EVERY SINGLE input,
    #we will update our weights. This is not always (maybe not even here) possible or practical, why?
    #Also, normally forward doesn't take in labels. Since we'll have forward call backprop, it'll take in labels
    #YOUR CODE HERE
    def forward(self, x, label):
        hidden_in = x.dot(self.weight1) + bias1
        hidden_act = self.relu(hidden_in)
        hidden_out = hidden_act.dot(self.weight2) + self.bias2
        act = self.sigmoid(hidden_out)
        loss = self.loss(act,label)
        self.backdrop(x, hidden_in, hidden_act, hidden_out, act, loss)

    #implement backprop, might help to have a helper function update weights
    #Recommend you check out the youtube channel 3Blue1Brown and their video on backprop
    #YOUR CODE HERE
    def backprop(self, x, hidden_in, hidden_act, hidden_out, act, loss): #What else might we need to take in as arguments? Modify as necessary

        #Compute the gradients first
        #First will have to do with combining derivative of sigmoid, output layer, and what else?
        #np.sum(x, axis, keepdims) may be useful

        #Update your weights and biases. Use a learning rate of 0.1, and update on every call to backprop
        lr = .1
        loss_act = 1 if act>0 else -1
        act_hidden_out = self.sigmoid(hidden_out) * (1 - self.sigmoid(hidden_out))
        loss_weight2 = hidden_act.T.dot(loss_act * act_hidden_out)
        loss_bias2 = loss_act * act_hidden_out
        hidden_out_hidden_act = self.weight2
        hidden_act_hidden_in = int(hidden_in>0)
        hidden_in_weight1 = x
        loss_weight1 = hidden_in_weight1.T.dot(hidden_act_hidden_in * hidden_out_hidden_act.T.dot(loss_act * act_hidden_out))
        loss_bias1 = hidden_act_hidden_in * hidden_out_hidden_act.T.dot(loss_act * act_hidden_out)

        self.weight1 -= lr * loss_weight1
        self.weight2 -= lr * loss_weight2
        self.bias1 -= lr * loss_bias1
        self.bias2 -= lr * loss_bias2