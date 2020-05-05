import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

#Step 1. Create dataset
train_dataset=dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset=dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

#Step 2. Make it iterable
batch_size=100
iterations=3000
epochs=int(iterations/(len(train_dataset)/batch_size))

train_loader=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#Step 3. Create model class
class CNNModel(nn.Module):
	def __init__(self):
		super(CNNModel, self).__init__()

		self.cnn1=nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
		self.relu1=nn.ReLU()
		# self.avgpool1=nn.AvgPool2d(kernel_size=2)
		self.maxpool1=nn.MaxPool2d(kernel_size=2)

		self.cnn2=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
		self.relu2=nn.ReLU()
		self.maxpool2=nn.MaxPool2d(kernel_size=2)
		# self.avgpool2=nn.AvgPool2d(kernel_size=2)

		self.fc1=nn.Linear(32*4*4, 10)
		# Careful with padding - linear layer input size is dependent on padding size!
	def forward(self, x):
		out=self.cnn1(x)
		out=self.relu1(out)
		out=self.maxpool1(out)
		# out=self.avgpool1(out)

		out=self.cnn2(out)
		out=self.relu2(out)
		out=self.maxpool2(out)
		# out=self.avgpool2(out)

		out=out.view(out.size(0), -1)
		out=self.fc1(out)

		return out

#4. Instantiate model class
model=CNNModel()
if torch.cuda.is_available():
	model.cuda()

#5. Instantiate loss class
criterion=nn.CrossEntropyLoss()

#6. Instantiate optimizer class
learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)

#7. Train the model
#CNN input is (1, 28, 28), while FNN input was 1, 28*28
iter=0
for epoch in range(epochs):
	for i, (images, labels) in enumerate(train_loader):
		if torch.cuda.is_available():
			images=Variable(images).cuda()
			labels=Variable(labels).cuda()
		else:
			images=Variable(images)
			labels=Variable(labels)
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
					images=Variable(images).cuda()
				else:
					images=Variable(images)
				outputs=model(images)
				_, predicted=torch.max(outputs.data, 1)
				total+=labels.size(0)
				if torch.cuda.is_available():
					correct+=(predicted.cpu()==labels.cpu()).sum()
				else:
					correct+=(predicted==labels).sum()

			accuracy=100*correct//total
			print('Iteration {}, loss: {}, accuracy: {}%'.format(iter, loss.data, accuracy))
