import os
import argparse
from random import shuffle


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help='Specify the path to the directory that contains images')
parser.add_argument('--save-dir', type=str, help='Specify path to directory to save train, val txt files')
args = parser.parse_args()

def writePathToFile():
    print("Please ensure that your data has been moved to a folder named 'data' which has 2 subfolders named 1 for nightmare images and 0 for non nightmare images")
    all_file = open(os.path.join(args.save_dir, 'all_data.txt'), 'w')
    train_file = open(os.path.join(args.save_dir, 'train.txt'), 'w')
    val_file = open(os.path.join(args.save_dir, 'val.txt'), 'w')

    count = 10
    for dirpath, dirnames, filenames in os.walk(args.dir):
        if len(filenames) > 0:
            for fname in filenames:
                src = os.path.join(dirpath, fname)
                name, ext = os.path.splitext(src)
                dst = os.path.join(dirpath, str(count)+ext)
                print(src, dst)
                os.rename(src, dst)
                all_file.write(dst+', '+dirpath[-1])
                all_file.write('\n')
                count += 1

    all_file.close()
    with open('all_data.txt', 'r') as all:
        lines = all.readlines()
        num_lines = list(range(len(lines)))
        shuffle(num_lines)
        train_lines = num_lines[:int(0.8*len(num_lines))]
        val_lines = num_lines[int(0.8*len(num_lines)):]
        for index in train_lines:
            train_file.write(lines[index])
        for index in val_lines:
            val_file.write(lines[index])

    all.close()

if __name__ == '__main__':
    writePathToFile()
