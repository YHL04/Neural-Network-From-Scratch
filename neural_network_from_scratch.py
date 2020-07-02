import numpy as np                 # pip install numpy
import matplotlib.pyplot as plt    # pip install matplotlib

# two node linear neural network
class NN:

	def __init__(self, m=0, b=0):
		self.m = m
		self.b = b


	def forward(self, x):
		return self.m*x + self.b
	

	def loss_mse(self, true, pred):
		return np.square(true - pred)


	def backpropagation(self, x, y, batch_size=1, learning_rate=0.01):
		pred = self.forward(x)

		# calculate m and b gradient with respect to loss using partial derivatives
		# devide by batch size and sum to account for the whole batch
		dm = -2 * (y - self.forward(x))
		dm = np.sum(dm / batch_size)
		db = -2*x * (y - self.forward(x))
		db = np.sum(db / batch_size)


		self.m -= dm * learning_rate
		self.b -= db * learning_rate


		loss = np.average(self.loss_mse(y, pred))
		return loss


def generator(size=100, m=0.7, b=0.2):
	x = np.random.rand(size)
	y = m*x + b
	return x, y, size


LEARNING_RATE = 0.001
STEP = 1000

model = NN()

plt.xlabel('LOSS')
plt.ylabel('STEPS')

loss = []
for t in range(STEP):
	x, y, batch_size = generator()
	loss = model.backpropagation(x, y, batch_size=batch_size, learning_rate=LEARNING_RATE)

	print("Loss: ", loss)
	plt.scatter(t, loss)
	plt.pause(0.03)

print('Final Network: y = {}x + {}'.format(model.m, model.b))
plt.show()


