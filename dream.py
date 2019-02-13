import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display
import numpy as np
import torch, os, time
import scipy.ndimage as nd
from torch.autograd import Variable
import hyperparameters as hyp
import warnings
from tqdm import tqdm

def showarray(a, file_name, iterations, experiment_path, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    for img in a:
        if img.shape[0]==224:
            PIL.Image.fromarray(img).save(experiment_path+"/iteration_{}/{}".format(iterations, file_name), fmt)
            file_name=file_name+1
    

def showtensor(a, file_name, iterations, experiment_path):
    mean = np.tile(np.array([0.485, 0.456, 0.406]).reshape([1, 1, 1, 3]), [a.shape[0],1,1,1])
    std = np.tile(np.array([0.229, 0.224, 0.225]).reshape([1, 1, 1, 3]), [a.shape[0],1,1,1])
    inp = a
    inp = inp.transpose(0, 2, 3, 1)
    inp = std * inp + mean
    inp *= 255
    showarray(inp, file_name, iterations, experiment_path)

def objective_L2(dst, guide_features):
    return dst.data

def make_step(base_img, img, model, file_name, experiment_path, control=None, distance=objective_L2):
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
    
    learning_rate = 2e-2
    max_jitter = 32
    num_iterations = hyp.ITERATIONS
    show_every = 10
    guide_features = control
    iteration_stack = []

    for i in range(num_iterations):
        shift_x, shift_y = np.random.randint(-max_jitter, max_jitter + 1, 2)
        img = np.roll(np.roll(img, shift_x, -1), shift_y, -2)
        # apply jitter shift
        model.zero_grad()
        img_tensor = torch.Tensor(img)
        if torch.cuda.is_available():
            img_variable = Variable(img_tensor.cuda(), requires_grad=True)
        else:
            img_variable = Variable(img_tensor, requires_grad=True)

        act_value, classification = model.resnet(img_variable, hyp.END_LAYER)
        diff_out = distance(act_value, guide_features)
        act_value.backward(diff_out)
        ratio = np.abs(img_variable.grad.data.cpu().numpy()).mean()
        learning_rate_use = learning_rate / ratio
        img_variable.data.add_(img_variable.grad.data * learning_rate_use)
        img = img_variable.data.cpu().numpy()  # b, c, h, w
        img = np.roll(np.roll(img, -shift_x, -1), -shift_y, -2)
        img[0, :, :, :] = np.clip(img[0, :, :, :], -mean / std,
                                  (1 - mean) / std)
        if i == 0 or (i + 1) % show_every == 0:
            if img.shape[2]==224:
                iteration_stack.append(img)
                showtensor(np.concatenate((base_img,img),axis=3), file_name, i+1, experiment_path)
            else:
                showtensor(img, file_name, i+1, experiment_path)
    return img, iteration_stack, classification


def dream(model,
          base_img,
          file_name,
          experiment_path,
          octave_n=6,
          octave_scale=1.4,
          control=None,
          distance=objective_L2):
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')
    octaves = [base_img]
    for i in range(octave_n - 1):
        octaves.append(
            nd.zoom(
                octaves[-1], (1, 1, 1.0 / octave_scale, 1.0 / octave_scale),
                order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(
                detail, (1, 1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        input_oct = octave_base + detail
        out, iteration_stack, classification = make_step(base_img, input_oct, model, file_name, experiment_path, control, distance=distance)
        detail = out - octave_base
    return iteration_stack, classification