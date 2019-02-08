import torchvision.models as models
import torch.nn as nn
import hyperparameters as hyp
import torch

class Model(nn.Module):
	def __init__(self):
		super().__init__()
		if hyp.DEPTH==50:
			self.base_model = models.resnet50(pretrained=True)
		else:
			if hyp.DEPTH==100:
				self.base_model = models.resnet101(pretrained=True)
			else:
				if hyp.DEPTH==150:
					self.base_model = models.resnet152(pretrained=True)
		self.classification_layer = nn.Sequential(	nn.ReLU(), nn.Linear(1000,500), nn.ReLU(),  nn.Linear(500,2))

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

	def train(self):
		self.base_model.train()

	def eval(self):
		self.base_model.eval()