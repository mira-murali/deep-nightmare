from dream import dream, objective_L2
from model import Model
from train import train
import numpy as np
from dataloader import get_loader
import hyperparameters as hyp
import torch, sys, os, time
from tqdm import tqdm
from itertools import product
import PIL.Image


def save_images(album, file_name, experiment_path):
    mean = np.tile(np.array([0.485, 0.456, 0.406]).reshape([1, 1, 1, 3]), [album.shape[0],1,1,1])
    std = np.tile(np.array([0.229, 0.224, 0.225]).reshape([1, 1, 1, 3]), [album.shape[0],1,1,1])
    inp = album.transpose(0, 2, 3, 1)
    inp = std * inp + mean
    inp *= 255
    a = np.uint8(np.clip(inp, 0, 255))
    for img in a:
            PIL.Image.fromarray(img).save(experiment_path+"/{}".format(file_name), "jpeg")
            file_name=file_name+1


def trip(trained_model=None):
	if trained_model is None:
		trained_model, training_dir = train(Model().cuda())
	else:
		path_to_model = trained_model
		trained_model = Model()
		trained_model.load_state_dict(torch.load(path_to_model))
		trained_model = trained_model.cuda()
		try:
			training_dir = path_to_model[:-path_to_model[::-1].index("/")]
		except ValueError:
			sys.exit("Model should be inside a directory !")
	experiment_path = training_dir+'experiment_{}'.format(time.time())
	os.mkdir(experiment_path)
	for OCTAVES in hyp.OCTAVES:
		octave_path = experiment_path+"/octave_{}".format(OCTAVES)
		os.mkdir(octave_path)
		os.mkdir(octave_path+"/iteration_{}".format(1))
		for itr in range(1,hyp.ITERATIONS//10+1):
			os.mkdir(octave_path+"/iteration_{}".format(itr*10))
	test_loader = get_loader(loader="test")
	



	file_name=1
	for image,__ in tqdm(test_loader):
		octave_stack=[]
		for OCTAVES in hyp.OCTAVES:
			iteration_stack = dream(trained_model,
									image.numpy(),
									file_name,
									experiment_path+"/octave_{}".format(OCTAVES),
									octave_n=OCTAVES,
									octave_scale=1.4,
									control=None,
									distance=objective_L2)
			octave_stack.append(np.concatenate(iteration_stack, axis=3))
		album = np.concatenate((np.tile(image.numpy(),[1,1,len(hyp.OCTAVES),1]),np.concatenate(octave_stack, axis=2)), axis=3)
		save_images(album, file_name, experiment_path)
		file_name=file_name+image.shape[0]






if __name__ == "__main__":
	trip("training_ResNet150_1549615892.251924/checkpoint_42.pth")