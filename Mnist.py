import csv
import numpy as np
import NeuralNetwork as net
import time


def ReadCSV(filename):
	print("\nLoading " + filename + " ...")

	inputs = []
	targets = []

	with open(filename) as csvfile:
		reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			y = np.zeros(10) + 0.01
			y[int(row[0])] = 0.99
			targets.append(y)
			inputs.append((np.asfarray(row[1:])/255)*0.99 + 0.01)

	return np.asarray(inputs), np.asarray(targets)

inputs,targets = ReadCSV("mnist_train.csv")

X_nodes = 784
L1_nodes = 200
Y_nodes = 10
epochs = 10
lr = 0.001

nn = net.NeuralNetwork(X_nodes,L1_nodes,Y_nodes,lr)

number_examples = len(inputs)

print("\nTraining...")

for e in range(epochs):
	init_time = time.time()
	err = 0

	for i in range(number_examples):
		error = nn.Train(inputs[i],targets[i])
		err = err + error
	err = err/number_examples
	finish_time = time.time()
	diff = round((finish_time - init_time),2)
	time_to_finish = round(((epochs - e)*diff)/60,2)
	print("Error: " + str(err) + " | EPOCH: " + str(e) + " | Time to finish: " + str(time_to_finish) + " mins")

#nn.Save("MNIST")
#print("Weights and biases saved.")

x,y = ReadCSV("mnist_test.csv")

test_examples = len(x)

ok_predictions = 0

for i in range(test_examples):
	expected = np.argmax(y[i])
	prediction = np.argmax(nn.Query(x[i]))
	if expected==prediction:
		ok_predictions += 1

accuracy = round((ok_predictions/test_examples)*100,2)
print("Accuracy on test data: " + str(accuracy) + "%")

