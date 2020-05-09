import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

#1. Load the dastasets
train_dataset=dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset=dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

#2. Make them iterable
batch_size=100
iterations=3000
num_epochs=iterations/(len(train_dataset)/batch_size)
num_epochs=int(num_epochs)

train_loader=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#3. Create model class
class RNNModel(nn.Module):
	def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
		super(RNNModel, self).__init__()
		self.hidden_dim=hidden_dim
		self.layer_dim=layer_dim
		self.rnn=nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh') # relu is causing weird issues here
		# batch_first causes the tensor to be in shape of: (batch_dim, seq_dim, input_dim)

		# readout layer
		self.fc=nn.Linear(hidden_dim, output_dim)
	def forward(self, x):
		# We have to make a variable which corresponds to the hidden state
		h0=Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).cuda()
		out, hn=self.rnn(x, h0)
		out=self.fc(out[:, -1, :]) # we're taking the last iteration, the last time step.
		# This causes one "forward()" to perform as many time steps as the input requires.
		return out

#4. Instantiate Model Class
input_dim=28 # only because our input images are of size 28*28
hidden_dim=100
layer_dim=2
output_dim=10 # again, only because there are 10 labels we consider

model=RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
if torch.cuda.is_available():
	model.cuda()
#5. Instantiate loss class
criterion=nn.CrossEntropyLoss()

#6. Instantiate optimizer class
learning_rate=0.1
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
# There are six parameters in this moment: two "ordinary" linear function and one
# recurrent linear function, each of which has two parameters: A and B

# 7. Train model

# number of time steps to unroll
seq_dim=28

iter=0
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		# if torch.cuda.is_available():
		images=Variable(images.view(-1, seq_dim, input_dim)).cuda() # because batch_first==True	
		labels=Variable(labels).cuda()
		# else:
		# 	images=Variable(images.view(-1, seq_dim, input_dim)) # because batch_first==True
		# 	labels=Variable(labels)
		optimizer.zero_grad()
		outputs=model(images)
		loss=criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		iter+=1
		if iter%500==0:
			correct=0
			total=0
			for images, labels in test_loader:
				if torch.cuda.is_available():
					images=Variable(images.view(-1, seq_dim, input_dim)).cuda()
				outputs=model(images)
				_, predicted=torch.max(outputs.data, 1)
				total+=labels.size(0)
				correct+=(predicted.cpu()==labels.cpu()).sum()

			accuracy=100*correct//total
			print('Iteration {}, loss: {}, accuracy: {}%'.format(iter, loss.data, accuracy))