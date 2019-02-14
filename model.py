import torchvision.models as models
import torch.nn as nn
import hyperparameters as hyp
import torch

class Model(nn.Module):
	def __init__(self):
		super().__init__()
		if hyp.DEPTH == 20:
			self.base_model = models.resnet18(pretrained=hyp.pretrained)
		elif hyp.DEPTH == 30:
			self.base_model = models.resnet34(pretrained=hyp.pretrained)
		elif hyp.DEPTH==50:
			self.base_model = models.resnet50(pretrained=hyp.pretrained)
		elif hyp.DEPTH == 100:
			self.base_model = models.resnet101(pretrained=hyp.pretrained)
		elif hyp.DEPTH == 150:
			self.base_model = models.resnet152(pretrained=hyp.pretrained)
#		self.classification_layer = nn.Sequential(	nn.ReLU(), nn.Linear(1000,500), nn.ReLU(),  nn.Linear(500,2))
		self.classification_layer = nn.Sequential(	nn.ReLU(), nn.Linear(1000,2))

	def forward(self, input_image):
		"""
		input_image : B x 3 x 224 x 224
		"""
		extracted_features = self.base_model.forward(input_image)
		output = self.classification_layer(extracted_features)
		"""
		output : 1=Hellscape, 0=Normal
		"""
		return output

	def resnet(self, input_image, end_layer):
		"""
		end_layer range from 1 to 4
		"""
		x = self.base_model.conv1(input_image)
		x = self.base_model.bn1(x)
		x = self.base_model.relu(x)
		x = self.base_model.maxpool(x)

		layers = [self.base_model.layer1, self.base_model.layer2, self.base_model.layer3, self.base_model.layer4]
		for i in range(end_layer):
			x = layers[i](x)
		"""
		with torch.no_grad():
			output = self.forward(input_image)
		return x, output.max(-1)[-1]
		"""
		return x, None

	def train(self):
		self.base_model.train()

	def eval(self):
		self.base_model.eval()
