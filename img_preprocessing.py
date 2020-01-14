import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from tqdm import tqdm  # _notebook as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import os
from scipy.optimize import minimize
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, utils

from math import sin, cos
from config import Config
from math import floor

# From camera.zip
camera_matrix = np.array([[2304.5479, 0, 1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)


# def imread(path, mask=False, fast_mode=False):
#     img = cv2.imread(path)
#     if mask:
#         imagemask = cv2.imread(path, 0)
#         try:
#             imagemaskinv = cv2.bitwise_not(imagemask)
#             res = cv2.bitwise_and(img, img, mask=imagemaskinv)
#             img = res
#         except:
#             pass
#
#     if not fast_mode and img is not None and len(img.shape) == 3:
#         img = np.array(img[:, :, ::-1])
#     return img

def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img


def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x


def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords


def get_img_coords(s):
    '''
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image (row)
        ys: y coordinates in the image (column)
    '''
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2]  # z = Distance from the camera
    return img_xs, img_ys


def _regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict


def preprocess_image(img, flip=False):
    img = img[img.shape[0] // 2:]
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 6]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
    if flip:
        img = img[:, ::-1]
    return (img / 255).astype('float32')


DISTANCE_THRESH_CLEAR = 2

train = pd.read_csv(Config.DATA_PATH + 'train.csv')
test = pd.read_csv(Config.DATA_PATH + 'sample_submission.csv')

points_df = pd.DataFrame()
for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
    arr = []
    for ps in train['PredictionString']:
        coords = str2coords(ps)
        arr += [c[col] for c in coords]
    points_df[col] = arr

print('len(points_df)', len(points_df))
points_df.head()

xzy_slope = LinearRegression()
X = points_df[['x', 'z']]
y = points_df['y']
xzy_slope.fit(X, y)
print('MAE with x:', mean_absolute_error(y, xzy_slope.predict(X)))

img = imread(Config.DATA_PATH + 'train_images/ID_8a6e65317' + '.jpg')
IMG_SHAPE = img.shape


def convert_3d_to_2d(x, y, z, fx=2304.5479, fy=2305.8757, cx=1686.2379, cy=1354.9849):
    # stolen from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline
    return x * fx / z + cx, y * fy / z + cy


def optimize_xy(r, c, x0, y0, z0, flipped=False):
    def distance_fn(xyz):
        x, y, z = xyz
        xx = -x if flipped else x
        slope_err = (xzy_slope.predict([[xx, z]])[0] - y) ** 2
        x, y = convert_3d_to_2d(x, y, z)
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * Config.IMG_HEIGHT / (IMG_SHAPE[0] // 2) / Config.MODEL_SCALE
        y = (y + IMG_SHAPE[1] // 6) * Config.IMG_WIDTH / (IMG_SHAPE[1] * 4 / 3) / Config.MODEL_SCALE
        return max(0.2, (x - r) ** 2 + (y - c) ** 2) + max(0.4, slope_err)

    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z_new


def _regr_back(regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)

    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict


def clear_duplicates(coords):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2) ** 2).sum())
            if distance < DISTANCE_THRESH_CLEAR:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]


def extract_coords(prediction, flipped=False, thr=0.0):
    logits = prediction[0]
    regr_output = prediction[1:]
    if Config.USE_GAUSSIAN:
        logits = _nms(torch.tensor(logits).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).data.cpu().numpy()

    points = np.argwhere(logits > thr)
    # print("points_len:{}".format(len(points)))

    col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
    coords = []
    for r, c in points:
        regr_dict = dict(zip(col_names, regr_output[:, r, c]))
        coords.append(_regr_back(regr_dict))
        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = \
            optimize_xy(r, c,
                        coords[-1]['x'],
                        coords[-1]['y'],
                        coords[-1]['z'], flipped)
    coords = clear_duplicates(coords)
    return coords


def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)


def get_mask_and_regr(img, labels, flip=False):
    mask = np.zeros([Config.IMG_HEIGHT // Config.MODEL_SCALE, Config.IMG_WIDTH // Config.MODEL_SCALE], dtype='float32')
    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    regr = np.zeros(
        [Config.IMG_HEIGHT // Config.MODEL_SCALE, Config.IMG_WIDTH // Config.MODEL_SCALE, Config.N_CLASS - 1],
        dtype='float32')
    coords = str2coords(labels)
    xs, ys = get_img_coords(labels)

    if not Config.USE_GAUSSIAN:
        for x, y, regr_dict in zip(xs, ys, coords):
            x, y = y, x
            x = (x - img.shape[0] // 2) * Config.IMG_HEIGHT / (img.shape[0] // 2) / Config.MODEL_SCALE
            x = np.round(x).astype('int')
            y = (y + img.shape[1] // 6) * Config.IMG_WIDTH / (img.shape[1] * 4 / 3) / Config.MODEL_SCALE
            y = np.round(y).astype('int')
            if 0 <= x < Config.IMG_HEIGHT // Config.MODEL_SCALE and 0 <= y < Config.IMG_WIDTH // Config.MODEL_SCALE:
                mask[x, y] = 1
                regr_dict = _regr_preprocess(regr_dict, flip)
                regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]
    else:
        mask = heatmap(ys, xs, Config.IMG_HEIGHT // Config.MODEL_SCALE, Config.IMG_WIDTH // Config.MODEL_SCALE)[:, :, 0]
        mask = mask.astype(np.float32)

        for x, y, regr_dict in zip(xs, ys, coords):
            x, y = y, x
            x = (x - img.shape[0] // 2) * Config.IMG_HEIGHT / (img.shape[0] // 2) / Config.MODEL_SCALE
            x = np.round(x).astype('int')
            y = (y + img.shape[1] // 6) * Config.IMG_WIDTH / (img.shape[1] * 4 / 3) / Config.MODEL_SCALE
            y = np.round(y).astype('int')
            if 0 <= x < Config.IMG_HEIGHT // Config.MODEL_SCALE and 0 <= y < Config.IMG_WIDTH // Config.MODEL_SCALE:
                regr_dict = _regr_preprocess(regr_dict, flip)
                regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]

    if flip:
        mask = np.array(mask[:, ::-1])
        regr = np.array(regr[:, ::-1])
    return mask, regr


def heatmap(u, v, output_height=128, output_width=128, sigma=1):
    def get_heatmap(p_x, p_y):
        X1 = np.linspace(1, output_width, output_width)
        Y1 = np.linspace(1, output_height, output_height)
        [X, Y] = np.meshgrid(X1, Y1)
        X = X - floor(p_x)
        Y = Y - floor(p_y)
        D2 = X * X + Y * Y
        E2 = 2.0 * sigma ** 2
        Exponent = D2 / E2
        heatmap = np.exp(-Exponent)
        heatmap = heatmap[:, :, np.newaxis]
        return heatmap

    output = np.zeros((output_height, output_width, 1))
    for i in range(len(u)):
        x = (u[i] - img.shape[0] // 2) * Config.IMG_HEIGHT / (img.shape[0] // 2) / Config.MODEL_SCALE
        x = np.round(x).astype('int')
        y = (v[i] + img.shape[1] // 6) * Config.IMG_WIDTH / (img.shape[1] * 4 / 3) / Config.MODEL_SCALE
        y = np.round(y).astype('int')

        heatmap = get_heatmap(y + 1, x + 1)
        output[:, :] = np.maximum(output[:, :], heatmap[:, :])

    return output


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def CreateMaskImages(imageName):
    trainimage = cv2.imread(Config.DATA_PATH + "/train_images/" + imageName + '.jpg')
    imagemask = cv2.imread(Config.DATA_PATH + "/train_masks/" + imageName + ".jpg", 0)
    try:
        imagemaskinv = cv2.bitwise_not(imagemask)
        res = cv2.bitwise_and(trainimage, trainimage, mask=imagemaskinv)
        res = trainimage - res / 4
        res = np.round(res).astype(np.uint8)
        b, g, r = cv2.split(res)
        res = cv2.merge([r, g, b])

        # res = cv2.add(trainimage, np.zeros(np.shape(trainimage), dtype=np.uint8), mask=imagemaskinv)

        # cut upper half,because it doesn't contain cars.
        res = res[res.shape[0] // 2:]

        return res
    except:
        trainimage = trainimage[trainimage.shape[0] // 2:]
        return trainimage


if __name__ == '__main__':
    train = pd.read_csv(Config.DATA_PATH + 'train.csv')

    trainimage = cv2.imread(Config.DATA_PATH + "/train_images/" + train['ImageId'][0] + '.jpg')
    imagemask = cv2.imread(Config.DATA_PATH + "/train_masks/" + train['ImageId'][0] + ".jpg", 0)
    imagemaskinv = cv2.bitwise_not(imagemask)
    res = cv2.bitwise_and(trainimage, trainimage, mask=imagemaskinv)
    res[np.where(np.sum(res, axis=2) == 0)] = [255, 255, 255]
    res = np.array(res[:, :, ::-1])
    plt.imshow(res)
    plt.show()
    cv2.imwrite(Config.DATA_PATH + "MaskTest/" + train['ImageId'][0] + ".jpg", res)

    # trainImg = CreateMaskImages('ID_0a1cb53b1')
    #
    # plt.figure(figsize=(24, 24))
    # plt.title('mask image')
    # plt.imshow(trainImg)
    # plt.show()
    Config.USE_GAUSSIAN = True

    img0 = imread(Config.DATA_PATH + 'train_images/' + train['ImageId'][0] + '.jpg')
    img_ = preprocess_image(img0)
    #
    mask, regr = get_mask_and_regr(img0, train['PredictionString'][0])

    print('img.shape', img_.shape, 'std:', np.std(img_))
    print('mask.shape', mask.shape, 'std:', np.std(mask))
    print('regr.shape', regr.shape, 'std:', np.std(regr))

    Config.USE_GAUSSIAN = False

    mask1, _ = get_mask_and_regr(img0, train['PredictionString'][0])

    #
    # plt.figure(figsize=(16, 16))
    # plt.title('Processed image')
    # plt.imshow(img)
    # plt.show()
    #
    # plt.figure(figsize=(16, 16))
    # plt.title('heatmap')
    # plt.imshow(mask)
    # plt.show()
    #
    #
    # plt.figure(figsize=(16, 16))
    # plt.title('Detection Mask')
    # plt.imshow(mask1)
    # plt.show()

    #
    # plt.figure(figsize=(16, 16))
    # plt.title('Yaw values')
    # plt.imshow(regr[:, :, -2])
    # plt.show()
