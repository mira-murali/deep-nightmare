import torch 
import torch.optim as optim
import torch.cuda as cuda
import numpy as np
from tqdm import tqdm

def train(model, train_loader):
	model.train()
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	if cuda.is_available():
		loss = nn.CrossEntropyLoss().cuda()
	else:
		loss = nn.CrossEntropyLoss()
	epoch = 0
	store_epoch_loss = [] 
	while True:
		epoch = epoch + 1
		epoch_loss = 0 
		store_batch_loss = []
		for image, label, batch_num in tqdm(enumerate(train_loader)):
			optimizer.zero_grad()
			prediction = model.forward(image)
			batch_loss = loss(prediction, label)
			batch_loss.backward()
			optimizer.step()
			store_batch_loss.append(batch_loss.clone().detatch().cpu())
			epoch_loss = np.mean(store_batch_loss)
			store_epoch_loss.append(epoch_loss)
		torch.save(model.state_dict(), path+"/model.pth")



		#Validate and auto-stop when overfitting
	return model