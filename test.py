from dream import dream, objective_L2
from model import Model
import train
import numpy as np
from dataloader import get_loader
import hyperparameters as hyp
import torch, sys, os, time
from tqdm import tqdm
from itertools import product
import PIL.Image
from utils import save_images



def trip(trained_model=None):
	if trained_model is None:
		trained_model, training_dir = train.train(Model().cuda())
		training_dir = training_dir+'/'
	else:
		path_to_model = trained_model
		trained_model = Model()
		trained_model.load_state_dict(torch.load(path_to_model))
		trained_model = trained_model.cuda()
		try:
			training_dir = path_to_model[:-path_to_model[::-1].index("/")]
		except ValueError:
			sys.exit("Model should be inside a directory !")
	print("Generating experiment...")
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
			iteration_stack, classification = dream(trained_model,
									image.numpy(),
									file_name,
									experiment_path+"/octave_{}".format(OCTAVES),
									octave_n=OCTAVES,
									octave_scale=1.4,
									control=None,
									distance=objective_L2)
			octave_stack.append(np.concatenate(iteration_stack, axis=3))
		album = np.concatenate((np.tile(image.numpy(),[1,1,len(hyp.OCTAVES),1]),np.concatenate(octave_stack, axis=2)), axis=3)
		save_images(album, file_name, classification, experiment_path)
		file_name=file_name+image.shape[0]






if __name__ == "__main__":
	if len(sys.argv)==1:
		trip()
	else:
		trip(sys.argv[1])
