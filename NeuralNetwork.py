import numpy as np

class NeuralNetwork:

	def __init__(self,X_nodes,L1_nodes,Y_nodes,learning_rate):
		self.Xnodes = X_nodes
		self.L1nodes = L1_nodes
		self.Ynodes = Y_nodes
		self.alfa = learning_rate

		np.random.seed(2231)

		self.W_XL1 = np.random.rand(self.Xnodes,self.L1nodes)
		self.W_L1Y = np.random.rand(self.L1nodes,self.Ynodes)

		self.B1 = np.random.rand(self.L1nodes,1)
		self.B2 = np.random.rand(self.Ynodes,1)

		print("W_XL1: \n" + str(self.W_XL1))
		print("W_L1Y: \n" + str(self.W_L1Y))
		print()
		print("B1: \n" + str(self.B1))
		print("B2: \n" + str(self.B2))

	def sigmoid(self,x):
		return 1/(1+np.exp(-x))

	def sigmoid_deriv(self,x):
		return self.sigmoid(x)*(1-self.sigmoid(x))	

	def relu(self,x):
		c = np.zeros_like(x)
		slope = 0.01
		c[x>0] = x[x>0]
		c[x<=0] = slope*x[x<=0]
		return c

	def relu_deriv(self,x):
		x[x<=0] = 0.01
		x[x>0] = 1.0
		return x

	def query(self,xlist):
		X = np.array(xlist, ndmin=2).T

		Z1 = np.dot(self.W_XL1.T,X) + self.B1
		A1 = self.relu(Z1)

		Z2 = np.dot(self.W_L1Y.T,A1) + self.B2
		Y = self.sigmoid(Z2)

		return Y


	def train(self,xlist,tlist):	
		X = np.array(xlist, ndmin=2).T
		T = np.array(tlist, ndmin=2).T

		#print("X:\n" + str(X))
		#print("T:\n" + str(T))

		############################## FEEDFORWARD ################################

		Z1 = np.dot(self.W_XL1.T,X) + self.B1
		A1 = self.relu(Z1)

		Z2 = np.dot(self.W_L1Y.T,A1) + self.B2
		Y = self.sigmoid(Z2)

		############################### ERRORS ####################################

		E_Y = T - Y
		E_W1 = np.dot(self.W_L1Y,E_Y)

		############################### DELTAS ###################################

		delta_W_L1Y = np.dot(-self.alfa*A1,(E_Y*self.sigmoid_deriv(Y)).T)
		delta_B2 = -self.alfa*E_Y*self.sigmoid_deriv(Y)

		delta_W_XL1 = np.dot(-self.alfa*X,(E_W1*self.relu_deriv(A1)).T)
		delta_B1 = -self.alfa*E_W1*self.relu_deriv(A1)
       
        	############################## UPDATES ###################################
		
		self.W_L1Y = self.W_L1Y - delta_W_L1Y
		self.B2 = self.B2 - delta_B2

		self.W_XL1 = self.W_XL1 - delta_W_XL1
		self.B1 = self.B1 - delta_B1

		return np.sum(np.absolute(E_Y))

	def save(self,name):
		np.savetxt(name+'_W_XL1.txt', self.W_XL1, fmt='%f')
		np.savetxt(name+'_W_L1Y.txt', self.W_L1Y, fmt='%f')
		np.savetxt(name+'_B1.txt', self.B1, fmt='%f')
		np.savetxt(name+'_B2.txt', self.B2, fmt='%f')
