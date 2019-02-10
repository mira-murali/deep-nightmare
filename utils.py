'''
Code run example:
python util.py --data-dir <path-to-folder-containing-images>

For instance, I have downloaded and extracted the five folders to a folder named 'images' under documents:
python util.py --data-dir /home/mira/Documents/images/
e.g.,
images/
    Class A Nightmare/
    Class A Non-Nightmare/
    Class B Nightmare/
    Class B Non-Nightmare/
    Final Testing Images/

Please ensure that the path you send to data-dir includes the parent directory of the five folders
If the directory name contains spaces, e.g., 'Art ML Images', please remember to include a '\' before each space when sending in the path:

python util.py --data-dir /home/mira/Documents/Art\ ML\ Images/
'''

import os
import argparse
import random
import glob
import shutil
import io
import numpy as np
import PIL.Image

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='./', type=str, help='Specify path to folder containing images')
args = parser.parse_args()

os.environ['CURRENT'] = os.getcwd()
os.environ['DATA'] = ''

def modify_folder():
    '''
    function to rename the folders of the images and move it to a new folder data which has the following structure:
    data/
        train/
            nightmare/
            notnightmare/
        test/
    '''
    if os.path.isdir(os.path.join(os.environ['CURRENT'], 'data')):
        shutil.rmtree(os.path.join(os.environ['CURRENT'], 'data'))
    os.mkdir(os.path.join(os.environ['CURRENT'], 'data'))
    os.environ['DATA'] = os.path.join(os.environ['CURRENT'], 'data')

    data_dir = os.path.abspath(args.data_dir)
    dirs = glob.glob(os.path.join(data_dir, '*'))
    if not os.path.isdir(os.path.join(os.environ['DATA'], 'train')):
        os.mkdir(os.path.join(os.environ['DATA'], 'train'))

    for dir in dirs:
        if 'Class A Nightmare' in dir:
            src = dir
            shutil.copytree(src, os.path.join(os.environ['DATA'], 'train', 'nightmare'))
        elif 'Class A Non-Nightmare' in dir:
            src = dir
            shutil.copytree(src, os.path.join(os.environ['DATA'], 'train', 'notnightmare'))
        elif 'Final Testing Images' in dir:
            src = dir
            shutil.copytree(src, os.path.join(os.environ['DATA'], 'test'))

def writePathToFile():
    '''
    function to rename the images in both train and test folders and write the paths into train, val, and test txt files
    '''

    all_file = open(os.path.join(os.environ['CURRENT'], 'all_data.txt'), 'w')
    train_file = open(os.path.join(os.environ['CURRENT'], 'train.txt'), 'w')
    val_file = open(os.path.join(os.environ['CURRENT'], 'val.txt'), 'w')
    test_file = open(os.path.join(os.environ['CURRENT'], 'test.txt'), 'w')

    directories = glob.glob(os.path.join(os.environ['DATA'], 'train/*'))

    for dir in directories:
        images = glob.glob(os.path.join(dir, '*'))
        images.sort()
        label = ""
        line = 1
        count = 1
        for image in images:
            src = image
            name, ext = os.path.splitext(image)
            dst = os.path.join(dir, str(count)+ext)
            while os.path.isfile(dst):
                count += 1
                dst = os.path.join(dir, str(count)+ext)
            os.rename(src, dst)
            if dst.find('/nightmare') < 0:
                all_file.write(dst+','+str(0))
                label = str(0)
            else:
                all_file.write(dst+','+str(1))
                label = str(1)
            all_file.write('\n')
            if (line%5)==0:
                val_file.write(dst+','+label+'\n')
            else:
                train_file.write(dst+','+label+'\n')

            line += 1
            count += 1


    all_file.close()
    val_file.close()
    train_file.close()
    with open('train.txt', 'r') as train:
        train_data = [(random.random(), line) for line in train]
    with open('val.txt', 'r') as val:
        val_data = [(random.random(), line) for line in val]

    train_data.sort()
    val_data.sort()
    with open('trainlist.txt', 'w') as new_train:
        for _, line in train_data:
            new_train.write(line)
    with open('vallist.txt', 'w') as new_val:
        for _, line in val_data:
            new_val.write(line)

    os.rename('trainlist.txt', 'train.txt')
    os.rename('vallist.txt', 'val.txt')

    train.close()
    val.close()
    new_train.close()
    new_val.close()

    count = 1
    test_images = glob.glob(os.path.join(os.environ['DATA'], 'test/*'))
    for test_image in test_images:
        src = test_image
        name, ext = os.path.splitext(test_image)
        dst = os.path.join(os.environ['DATA'], 'test', str(count)+ext)
        while os.path.isfile(dst):
            count += 1
            dst = os.path.join(os.environ['DATA'], 'test', str(count)+ext)
        os.rename(src, dst)
        test_file.write(dst+'\n')
        count += 1

    test_file.close()


def create_dir(name, parent=None):
	"""
	function to create directory at required depth
	if parent exists, ./parent/name is created else ./name is created
	You dont have to create parent
	TODO: Yuhan
	"""
	now=time.time()
	full_path=''
	if parent is not None and os.path.isdir('./'+parent):
		full_path = './'+parent+'/'+name
		os.mkdir(full_path)
		return full_path
	full_path = './'+name
	os.mkdir(full_path)
	return full_path

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



if __name__ == '__main__':
    modify_folder()
    writePathToFile()
