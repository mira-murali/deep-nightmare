import torch, os, time, signal, sys
import torch.optim as optim
import torch.cuda as cuda
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
from dataloader import get_loader
import torch.nn as nn
from model import Model
import matplotlib.pyplot as plt
import hyperparameters as hyp
from utils import save_images
from functools import partial
import test
from threading import Timer

def interrupt_handler(trained_model1, trained_model2):
	print("User input timed-out ! Using min loss...")
	test.trip(trained_model1)
	test.trip(trained_model2)
	print("Experiments generated ! Would you like to do another checkpoint? ")

def train(model):
	print("Training...")
	training_dir = "./training_{}_ResNet{}_{}".format(str(hyp.GRADES), hyp.DEPTH, time.time())
	os.mkdir(training_dir)
	os.mkdir(training_dir+'/misclassified')
	model.train()
	optimizer = optim.Adam(model.parameters(), lr=hyp.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	loss = nn.CrossEntropyLoss().cuda()
	train_loader = get_loader(loader='train', grades=hyp.GRADES, jitter=hyp.JITTER, animals=hyp.ANIMALS)
	epoch = 0
	store_epoch_loss = []
	store_epoch_loss_val = []
	store_epoch_acc_val = []
	try:
		for e in tqdm(range(hyp.EPOCHS)):
			epoch = e + 1
			epoch_loss = 0
			store_batch_loss = []
			for batch_num, (image, label) in enumerate(train_loader):
				optimizer.zero_grad()
				prediction = model.forward(image.cuda())
				batch_loss = loss(prediction, label.cuda())
				batch_loss.backward()
				optimizer.step()
				store_batch_loss.append(batch_loss.clone().cpu())
				epoch_loss = torch.FloatTensor(store_batch_loss).mean()
			store_epoch_loss.append(epoch_loss)
			torch.save(model.state_dict(), "{}/checkpoint_{}.pth".format(training_dir, epoch))
			plt.plot(store_epoch_loss[1:], label="Training Loss")

			model.eval()
			epoch_loss_val = 0
			epoch_acc_val = 0
			store_batch_loss_val = []
			store_batch_acc_val = []
			val_loader = get_loader(loader='val', grades=hyp.GRADES, animals=hyp.ANIMALS)
			misclassified_images = []
			for batch_num, (image, label) in enumerate(val_loader):
				with torch.no_grad():
					prediction = model.forward(image.cuda())
				batch_loss = loss(prediction, label.cuda())
				misclassified = prediction.max(-1)[-1].squeeze().cpu() != label
				misclassified_images.append(image[misclassified==1])
				batch_acc = misclassified.float().mean()
				store_batch_loss_val.append(batch_loss.cpu())
				store_batch_acc_val.append(batch_acc)
				epoch_loss_val = torch.FloatTensor(store_batch_loss_val).mean()
				epoch_acc_val = torch.FloatTensor(store_batch_acc_val).mean()
			store_epoch_loss_val.append(epoch_loss_val)
			store_epoch_acc_val.append(1-epoch_acc_val)
			plt.plot(store_epoch_loss_val[1:], label="Validation Loss")
			plt.plot(store_epoch_acc_val[1:], label="Validation Accuracy")
			plt.legend()
			plt.grid()
			plt.savefig("{}/Loss.png".format(training_dir))
			plt.close()
			misclassified_images = np.concatenate(misclassified_images,axis=0)
			validation_dir = training_dir+'/misclassified/checkpoint_{}'.format(epoch)
			os.mkdir(validation_dir)
			save_images(misclassified_images, 0, None, validation_dir)
			model.train()
		most_acc = max(store_epoch_acc_val)
		min_loss = min(store_epoch_loss_val)
		print("\nHighest accuracy of {} occured at {}%...Minimum loss occured at {}%...".format(most_acc, store_epoch_acc_val.index(most_acc)+1, store_epoch_loss_val.index(min_loss)+1))
		t = Timer(3*60, interrupt_handler, ["{}/checkpoint_{}.pth".format(training_dir, store_epoch_loss_val.index(min_loss)+1), "{}/checkpoint_{}.pth".format(training_dir, store_epoch_acc_val.index(most_acc)+1)])
		t.start()
		user_pick = input("Which checkpoint do you want to use ?\n")
		t.cancel()
		model.load_state_dict(torch.load("{}/checkpoint_{}.pth".format(training_dir, user_pick)))
	except KeyboardInterrupt:
		most_acc = max(store_epoch_acc_val)
		min_loss = min(store_epoch_loss_val)
		print("\nHighest accuracy of {} occured at {}%...Minimum loss occured at {}%...".format(most_acc, store_epoch_acc_val.index(most_acc)+1, store_epoch_loss_val.index(min_loss)+1))
		user_pick = input("Which checkpoint do you want to use ?\n")
		model.load_state_dict(torch.load("{}/checkpoint_{}.pth".format(training_dir, user_pick)))

	return model.cuda(), training_dir



if __name__ == "__main__":
	if len(sys.argv)==1:
		train(Model().cuda())
	else:
		mdl = Model()
		mdl.load_state_dict(torch.load(sys.argv[1]))
		train(mdl.cuda())
