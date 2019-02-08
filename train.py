import torch, os, time
import torch.optim as optim
import torch.cuda as cuda
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
from dataloader import get_loaders
import torch.nn as nn
from model import Model
import matplotlib.pyplot as plt


def train(model):
	training_dir = "./training_{}".format(time.time())
	os.mkdir(training_dir)
	model.train()
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	loss = nn.CrossEntropyLoss().cuda()
	train_loader, val_loader = get_loaders()
	epoch = 0
	store_epoch_loss = [] 
	store_epoch_loss_val = [] 
	store_epoch_acc_val = [] 
	while True:
		epoch = epoch + 1
		epoch_loss = 0 
		store_batch_loss = []
		for batch_num, (image, label) in tqdm(enumerate(train_loader)):
			optimizer.zero_grad()
			prediction = model.forward(image.cuda())
			batch_loss = loss(prediction, label.cuda())
			batch_loss.backward()
			optimizer.step()
			store_batch_loss.append(batch_loss.clone().cpu())
			epoch_loss = torch.FloatTensor(store_batch_loss).mean()
		store_epoch_loss.append(epoch_loss)
		torch.save(model.state_dict(), "{}/checkpoint_{}.pth".format(training_dir, epoch))
		plt.plot(store_epoch_loss, label="Training Loss")

		model.eval()
		epoch_loss_val = 0 
		epoch_acc_val = 0 
		store_batch_loss_val = []
		store_batch_acc_val = []
		for batch_num, (image, label) in tqdm(enumerate(val_loader)):
			with torch.no_grad():
				prediction = model.forward(image.cuda())
			batch_loss = loss(prediction, label.cuda())
			batch_acc = torch.abs(prediction.max(-1)[-1].squeeze().cpu()-label).float().mean()
			store_batch_loss_val.append(batch_loss.cpu())
			store_batch_acc_val.append(batch_acc)
			epoch_loss_val = torch.FloatTensor(store_batch_loss_val).mean()
			epoch_acc_val = torch.FloatTensor(store_batch_acc_val).mean()
		store_epoch_loss_val.append(epoch_loss_val)
		store_epoch_acc_val.append(epoch_acc_val)
		plt.plot(store_epoch_loss_val, label="Validation Loss")
		plt.plot(store_epoch_acc_val, label="Validation Accuracy")
		plt.legend()
		plt.savefig("{}/Loss.png".format(training_dir))
		plt.close()
		model.train()

		#Validate and auto-stop when overfitting
	return model


train(Model().cuda())