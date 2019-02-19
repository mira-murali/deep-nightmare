import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display
import numpy as np
import torch, os, time, sys
import scipy.ndimage as nd
from torch.autograd import Variable
import hyperparameters as hyp
import warnings
from tqdm import tqdm

def showarray(a):
    a = np.uint8(np.clip(a, 0, 255))
    for img in a:
        return img


def showtensor(a):
    mean = np.tile(np.array([0.485, 0.456, 0.406]).reshape([1, 1, 1, 3]), [a.shape[0],1,1,1])
    std = np.tile(np.array([0.229, 0.224, 0.225]).reshape([1, 1, 1, 3]), [a.shape[0],1,1,1])
    inp = a
    inp = inp.transpose(0, 2, 3, 1)
    inp = std * inp + mean
    inp *= 255
    return showarray(inp)

def objective_L2(dst, guide_features):
    return dst.data

def make_step(img, model, control=None, distance=objective_L2):
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])

    learning_rate = 2e-2
    max_jitter = 32
    num_iterations = np.random.randint(1,40)
    guide_features = control

    for i in range(num_iterations):
        shift_x, shift_y = np.random.randint(-max_jitter, max_jitter + 1, 2)
        img = np.roll(np.roll(img, shift_x, -1), shift_y, -2)
        # apply jitter shift
        model.zero_grad()
        img_tensor = torch.Tensor(img)
        img_variable = Variable(img_tensor.cuda(), requires_grad=True)

        act_value = model.resnet(img_variable, hyp.END_LAYER)[0]
        diff_out = distance(act_value, guide_features)
        act_value.backward(diff_out)
        ratio = np.abs(img_variable.grad.data.cpu().numpy()).mean()
        learning_rate_use = learning_rate / ratio
        img_variable.data.add_(img_variable.grad.data * learning_rate_use)
        img = img_variable.data.cpu().numpy()  # b, c, h, w
        img = np.roll(np.roll(img, -shift_x, -1), -shift_y, -2)
        img[0, :, :, :] = np.clip(img[0, :, :, :], -mean / std,
                                  (1 - mean) / std)
    return img


def dream(model, base_img):
    base_img = np.expand_dims(base_img.transpose(2,0,1), axis=0)
    octave_n    =   np.random.randint(3,9)
    octave_scale=   1.4
    control     =   None
    distance    =   objective_L2
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
        out = make_step(input_oct, model, control, distance=distance)
        detail = out - octave_base
    return showtensor(out)
