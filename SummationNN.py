# Build of the example seen at:
#https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6


import numpy as np

# Defines the size of the steps taken by weights towards minimizing loss function
learning_rate = 0.1

# Activation Function - Used to allow the neural net to be a NON-linear function
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

# Derivative of activation function - Used in training (backpropagation)
def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, input_data, labels):
        self.input = input_data
        # Weights for the first layer
        # Generates an array for each array in testing data (self.input.shape[1])
        # Initializes the weights randomly (Could train faster if weights were normally distributed)
        self.weights1 = np.random.rand(self.input.shape[1],4) 
        self.weights2 = np.random.rand(4,1)                 
        self.labels = labels
        self.output = np.zeros(self.labels.shape)

    # Passes the input data through the network 
    def feedforward(self):
        # Each Layer outputs the activation function(input * weights)
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def train(self):

        # Calculates derivative of loss function with respect to each weight (using the chain rule for composite functions)
        d_weights2 = np.dot(np.transpose(self.layer1), (2*(self.labels - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(np.transpose(self.input),  (np.dot(2*(self.labels - self.output) * sigmoid_derivative(self.output), np.transpose(self.weights2)) * sigmoid_derivative(self.layer1)))

        # Updates the weights towards minimizing the loss function by an amount specified by the learning rate
        # Smaller learning rate takes longer to train but can increase precision of model
        self.weights1 += d_weights1 * learning_rate
        self.weights2 += d_weights2 * learning_rate

    def changeData(self, input_data, output_data):
        self.input = input_data
        self.labels = output_data

# GOAL: 
# Multiplies each column by the amount given bellow and sums the resulting numbers
###   [[0.6], [0.3], [0.1]]

if __name__ == "__main__":
    # Sets up training data
    training_input_data = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    training_labels = np.array([[0.1],[0.4],[0.7],[1]])
    nn = NeuralNetwork(training_input_data, training_labels)

    # Training Loop - Re-using same data to make training dataset larger
    for i in range(100000):
        nn.feedforward()
        nn.train()

    # Testing data
    testing_input_data = np.array([[0,0,1],
                  [0,0,0],
                  [1,0,1],
                  [1,1,1]])

    # Labels for testing data (not really needed, just to see)
    testing_labels = np.array([[0.1],[0.0],[0.7],[1]])

    nn.changeData(testing_input_data, testing_labels)
    nn.feedforward()


    results = nn.output

    # Displays results
    # Model is overfitted due to the small amount of training data which had to be copied for training
    # Model performs poorly on data not seen
    # Model could be enhanced by adding layers and adding decimals to increase training data
    for i in range (results.shape[0]):
        print("________________")
        print("Output: ")
        print(results[i].item())
        print("Actual: ")
        print(testing_labels[i].item())
