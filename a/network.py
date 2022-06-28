import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class myVGG(nn.Module):

	def __init__(self, num_parameters):
		super(myVGG, self).__init__()

		vgg_base = models.vgg16(pretrained=True)
		
		self.features = vgg_base.features
		"""
		self.classifier = nn.Sequential(
				nn.Dropout(),
				nn.Linear(25088, 4096),
				nn.ReLU(inplace=True),
				nn.Dropout(),
				nn.Linear(4096, 4096),
				nn.ReLU(inplace=True),
				nn.Linear(4096, num_parameters),
		)
		"""
		self.classifier = vgg_base.classifier
		self.classifier._modules['6'] = nn.Linear(4096, num_parameters)
		#self.classifier._modules['7'] = nn.Tanh()

		"""
		for p in self.features.parameters():
			p.requires_grad = False
		"""
		


	def forward(self, x):

		
		x = self.features(x)
		x = x.view(x.size(0), -1)
		y = self.classifier(x)

		return y
		
		#return self.classifier._modules['6'](self.model(x))

class myVGG_bn(nn.Module):

	def __init__(self, num_parameters):
		super(myVGG_bn, self).__init__()

		vgg_base = models.vgg16_bn(pretrained=True)
		
		self.features = vgg_base.features
		"""
		self.classifier = nn.Sequential(
				nn.Dropout(),
				nn.Linear(25088, 4096),
				nn.ReLU(inplace=True),
				nn.Dropout(),
				nn.Linear(4096, 4096),
				nn.ReLU(inplace=True),
				nn.Linear(4096, num_parameters),
		)
		"""
		self.classifier = vgg_base.classifier
		self.classifier._modules['6'] = nn.Linear(4096, num_parameters)
		#self.classifier._modules['7'] = nn.Tanh()

		"""		
		for p in self.features.parameters():#freeze those weights
			p.requires_grad = False
		"""
		


	def forward(self, x):

		
		x = self.features(x)
		x = x.view(x.size(0), -1)
		y = self.classifier(x)

		return y
		
		#return self.classifier._modules['6'](self.model(x))

class myVGG19(nn.Module):

	def __init__(self, num_parameters):
		super(myVGG19, self).__init__()

		vgg19_base = models.vgg19(pretrained=True)
		
		self.features = vgg19_base.features

		self.classifier = vgg19_base.classifier
		self.classifier._modules['6'] = nn.Linear(4096, num_parameters)
		#self.classifier._modules['7'] = nn.Tanh()


	def forward(self, x):

		
		x = self.features(x)
		x = x.view(x.size(0), -1)
		y = self.classifier(x)

		return y
		
class myVGG19_bn(nn.Module):

	def __init__(self, num_parameters):
		super(myVGG19_bn, self).__init__()

		vgg19_base = models.vgg19_bn(pretrained=True)
		
		self.features = vgg19_base.features
		self.classifier = vgg19_base.classifier
		self.classifier._modules['6'] = nn.Linear(4096, num_parameters)
		#self.classifier._modules['7'] = nn.Tanh()

	def forward(self, x):

		x = self.features(x)
		x = x.view(x.size(0), -1)
		y = self.classifier(x)

		return y

class myAlexNet(nn.Module):

	def __init__(self, num_parameters):
		super(myAlexNet, self).__init__()

		alexNet_base = models.alexnet(pretrained=True)

		#print (alexNet_base)
		self.features = alexNet_base.features
		self.classifier = alexNet_base.classifier
		self.classifier._modules['6'] = nn.Linear(4096, num_parameters)
		#self.classifier._modules['7'] = nn.Tanh()

	def forward(self, x):

		x = self.features(x)
		x = x.view(x.size(0), -1)
		y = self.classifier(x)

		return y
		
		#return self.classifier._modules['6'](self.model(x))

class myResNet18(nn.Module):

	def __init__(self, num_parameters):
		super(myResNet18, self).__init__()

		ResNet_base = models.resnet18(pretrained=True)
		
		self.features = nn.Sequential(*list(ResNet_base.children())[:-1])
		self.fc = nn.Sequential( nn.Linear(512, num_parameters) )
		
	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		y = self.fc(x)

		return y
		


class tempModel(nn.Module):

	def __init__(self, num_parameters):
		super(tempModel, self).__init__()

		#input = 3 * 224 * 224

		self.conv1 = nn.Conv2d(3, 6, 5) 
		self.conv2 = nn.Conv2d(6, 16, 5) 
		self.fc1 = nn.Linear(16 * 53 * 53, 120)
		self.fc2 = nn.Linear(120, num_parameters)


	def forward(self, x):

		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # filter num1*110*110
		x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))# filter num2*53*53
		#x = x.view(-1, self.num_flat_features(x))
		print(x.size())
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)

		return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features




#print (myAlexNet(21))
#print 'network.py'
