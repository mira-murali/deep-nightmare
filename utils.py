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
    Class C Nightmare/
    Class C Non-Nightmare/
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


os.environ['CURRENT'] = os.getcwd()
if not os.path.isdir(os.path.join(os.environ['CURRENT'], 'data_files')):
    os.mkdir(os.path.join(os.environ['CURRENT'], 'data_files'))

os.environ['DATA'] = ''
os.environ['FILES'] = os.path.join(os.environ['CURRENT'], 'data_files')

def write_data(dir, file_name):
    dir_name = os.path.abspath(dir)
    labels = glob.glob(os.path.join(dir_name, '*'))
    all_file = open(os.path.join(os.environ['FILES'], file_name), 'w')
    class_to_label = {}
    count = 0
    for label in labels:
        class_to_label[label] = str(count)
        images = glob.glob(os.path.join(label, '*'))
        for image in images:
            all_file.write(image+','+class_to_label[label]+'\n')
        count += 1
    all_file.close()

def split_data(file_name, shuffle=True):
    if shuffle:
        shuffle_lines(file_name)
    last_slash = find_current_dirname(file_name)
    train_file = open(os.path.join(file_name[:-last_slash-1], 'train_animals.txt'), 'w')
    val_file = open(os.path.join(file_name[:-last_slash-1], 'val_animals.txt'), 'w')
    counter = 0
    with open(file_name) as src_file:
        f = src_file.readlines()
        counter = 0
        for line in f:
            if counter <= int(0.8*len(f)):
                train_file.write(line)
            else:
                val_file.write(line)
            counter += 1

def merge_files(file_list, new_file_name):
    out = open(new_file_name, 'w')
    for doc in file_list:
        lines = open(doc)
        for line in lines:
            out.write(line)

def remove_files(file_list):
    for doc in file_list:
        os.remove(doc)

def find_current_dirname(string):
    reverse_string = string[-1::-1]
    return reverse_string.find('/')

def shuffle_lines(file_name):
    with open(file_name, 'r') as file:
        file_data = [(random.random(), line) for line in file]
    file_data.sort()
    with open('temp.txt', 'w') as temp:
        for _, line in file_data:
            temp.write(line)
    os.rename('temp.txt', file_name)

def modify_folder():
    '''
    function to rename the folders of the images and move it to a new folder data which has the following structure:
    data/
        train/
            NightmareA/
            NightmareB/
            NightmareC/
            Non-NightmareA/
            Non-NightmareB/
            Non-NightmareC/
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

    grades = []
    for dir in dirs:
        if 'Final Testing Images' not in dir:
            last_slash = find_current_dirname(dir)
            split_words = dir[-last_slash:].split(' ')
            src = dir
            label = split_words[2]
            grade = split_words[1]
            grades.append(grade)
            shutil.copytree(src, os.path.join(os.environ['DATA'], 'train', label+grade))
        else:
            src = dir
            shutil.copytree(src, os.path.join(os.environ['DATA'], 'test'))
    grades = set(grades)
    return grades

def writePathToFile(grades):
    '''
    function to rename the images in both train and test folders and write the paths into train, val, and test txt files
    '''

    all_file = open(os.path.join(os.environ['FILES'], 'all_data.txt'), 'w')
    train_files = {}
    val_files = {}

    for grade in grades:
        train_files[(grade, '1')] = os.path.join(os.environ['CURRENT'], 'train'+grade+'1.txt')
        train_files[(grade, '0')] = os.path.join(os.environ['CURRENT'], 'train'+grade+'0.txt')
        val_files[(grade, '1')] = os.path.join(os.environ['CURRENT'], 'val'+grade+'1.txt')
        val_files[(grade, '0')] = os.path.join(os.environ['CURRENT'], 'val'+grade+'0.txt')

    test_file = open(os.path.join(os.environ['FILES'], 'test.txt'), 'w')

    directories = glob.glob(os.path.join(os.environ['DATA'], 'train/*'))

    train_file = ''
    val_file = ''
    for dir in directories:
        images = glob.glob(os.path.join(dir, '*'))
        images.sort()
        if 'Non-' in dir:
            label = '0'
        else:
            label = '1'
        line = 1
        count = 1
        split_word = dir[-1]
        train_file = open(train_files[(split_word, label)], 'w')
        val_file = open(val_files[(split_word, label)], 'w')

        for image in images:
            src = image
            name, ext = os.path.splitext(image)
            dst = os.path.join(dir, str(count)+ext)
            while os.path.isfile(dst):
                count += 1
                dst = os.path.join(dir, str(count)+ext)
            os.rename(src, dst)
            all_file.write(dst+','+label+'\n')
            if (line%5)==0:
                val_file.write(dst+','+label+'\n')
            else:
                train_file.write(dst+','+label+'\n')

            line += 1
            count += 1

        train_file.close()
        val_file.close()


    all_file.close()

    for grade in grades:

        train_list = [train_files[(grade, '0')], train_files[(grade, '1')]]
        val_list = [val_files[(grade, '0')], val_files[(grade, '1')]]

        train_name = os.path.join(os.environ['FILES'], 'train'+grade+'.txt')
        val_name = os.path.join(os.environ['FILES'], 'val'+grade+'.txt')

        merge_files(file_list=train_list, new_file_name=train_name)
        merge_files(file_list=val_list, new_file_name=val_name)

        remove_files(file_list=train_list)
        remove_files(file_list=val_list)

        shuffle_lines(train_name)
        shuffle_lines(val_name)

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

def save_images(album, file_name, classification, experiment_path):
    mean = np.tile(np.array([0.485, 0.456, 0.406]).reshape([1, 1, 1, 3]), [album.shape[0],1,1,1])
    std = np.tile(np.array([0.229, 0.224, 0.225]).reshape([1, 1, 1, 3]), [album.shape[0],1,1,1])
    inp = album.transpose(0, 2, 3, 1)
    inp = std * inp + mean
    inp *= 255
    a = np.uint8(np.clip(inp, 0, 255))
    for i,img in enumerate(a):
            if classification is None:
                PIL.Image.fromarray(img).save(experiment_path+"/{}.jpeg".format(file_name), "jpeg")
            else:
                PIL.Image.fromarray(img).save(experiment_path+"/{}_{}.jpeg".format(file_name, classification[i]), "jpeg")
            file_name=file_name+1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='images/', type=str, help='Specify path to folder containing images')
    parser.add_argument('--ANF', default=1, type=int, help='Specify if ANF dataset is being used')
    args = parser.parse_args()
    if not args.ANF:
        grades = modify_folder()
        writePathToFile(grades)
    else:
        write_data(args.data_dir, file_name = 'all_animals.txt')
        split_data(os.path.join(os.environ['FILES'], 'all_animals.txt'), shuffle=True)
    
