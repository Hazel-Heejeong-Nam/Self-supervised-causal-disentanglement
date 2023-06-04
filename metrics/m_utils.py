# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions that are useful for the different metrics."""
import numpy as np
import sklearn
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
import matplotlib.pyplot as plt
import os

import matplotlib.image as mpimg
import random
import math
import pandas as pd 
import torch
from PIL import Image
import torchvision.transforms as transforms
import subprocess
device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")


def generate_batch_factor_code(ground_truth_data, representation_function,
                               num_points, random_state, batch_size):
  """Sample a single training sample based on a mini-batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    num_points: Number of points to sample.
    random_state: Numpy random state used for randomness.
    batch_size: Batchsize to sample points.

  Returns:
    representations: Codes (num_codes, num_points)-np array.
    factors: Factors generating the codes (num_factors, num_points)-np array.
  """
  representations = None
  factors = None
  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_factors, current_observations = \
        ground_truth_data.sample(num_points_iter, random_state)
    if i == 0:
      factors = current_factors
      representations = representation_function(current_observations)
    else:
      factors = np.vstack((factors, current_factors))
      representations = np.vstack((representations,
                                   representation_function(
                                       current_observations)))
    i += num_points_iter
  return np.transpose(representations), np.transpose(factors)


def split_train_test(observations, train_percentage):
  """Splits observations into a train and test set.

  Args:
    observations: Observations to split in train and test. They can be the
      representation or the observed factors of variation. The shape is
      (num_dimensions, num_points) and the split is over the points.
    train_percentage: Fraction of observations to be used for training.

  Returns:
    observations_train: Observations to be used for training.
    observations_test: Observations to be used for testing.
  """
  num_labelled_samples = observations.shape[1]
  num_labelled_samples_train = int(
      np.ceil(num_labelled_samples * train_percentage))
  num_labelled_samples_test = num_labelled_samples - num_labelled_samples_train
  observations_train = observations[:, :num_labelled_samples_train]
  observations_test = observations[:, num_labelled_samples_train:]
  assert observations_test.shape[1] == num_labelled_samples_test, \
      "Wrong size of the test set."
  return observations_train, observations_test


def obtain_representation(observations, representation_function, batch_size):
  """"Obtain representations from observations.

  Args:
    observations: Observations for which we compute the representation.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Batch size to compute the representation.
  Returns:
    representations: Codes (num_codes, num_points)-Numpy array.
  """
  representations = None
  num_points = observations.shape[0] # num_variance_estimate
  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_observations = observations[i:i + num_points_iter]
    if i == 0:
      representations = representation_function(current_observations.to(device)).detach().cpu().numpy()
    else:
      representations = np.vstack((representations,representation_function(current_observations.to(device)).detach().cpu().numpy()))
    i += num_points_iter
  return np.transpose(representations)


def discrete_mutual_info(mus, ys):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
  return m


def discrete_entropy(ys):
  """Compute discrete mutual information."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
  return h



def make_discretizer(target, num_bins,
                     discretizer_fn):
  """Wrapper that creates discretizers."""
  return discretizer_fn(target, num_bins)


def _histogram_discretize(target, num_bins):
  """Discretization based on histograms."""
  discretized = np.zeros_like(target)
  for i in range(target.shape[0]):
    discretized[i, :] = np.digitize(target[i, :], np.histogram(
        target[i, :], num_bins)[1][:-1])
  return discretized


def normalize_data(data, mean=None, stddev=None):
  if mean is None:
    mean = np.mean(data, axis=1)
  if stddev is None:
    stddev = np.std(data, axis=1)
  return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev



def make_predictor_fn(predictor_fn):
  """Wrapper that creates classifiers."""
  return predictor_fn



def logistic_regression_cv():
  """Logistic regression with 5 folds cross validation."""
  return linear_model.LogisticRegressionCV(Cs=10,
                                           cv=model_selection.KFold(n_splits=5))


def gradient_boosting_classifier():
  """Default gradient boosting classifier."""
  return ensemble.GradientBoostingClassifier()



if not os.path.exists('/mnt/hazel/data/causal_data/pendulum/'): 
  os.makedirs('/mnt/hazel/data/causal_data/pendulum/train/')
  os.makedirs('/mnt/hazel/data/causal_data/pendulum/test/')

def projection(theta, phi, x, y, base = -0.5):
    b = y-x*math.tan(phi)
    shade = (base - b)/math.tan(phi)
    return shade
# 

def gen_pendulum_observation(factors):
  imageset = []
  for img in factors:
    i = img[0]
    j = img[1]
    shade = img[2]
    mid = img[3]
    
    
    fig = plt.figure(figsize=(1.0,1.0), dpi=96)
    theta = i*math.pi/200.0
    phi = j*math.pi/200.0
    x = 10 + 8*math.sin(theta)
    y = 10.5 - 8*math.cos(theta)

    ball = plt.Circle((x,y), 1.5, color = 'firebrick')
    gun = plt.Polygon(([10,10.5],[x,y]), color = 'black', linewidth = 3)

    light = projection(theta, phi, 10, 10.5, 20.5)
    sun = plt.Circle((light,20.5), 3, color = 'orange')
    shadow = plt.Polygon(([mid - shade/2.0, -0.5],[mid + shade/2.0, -0.5]), color = 'black', linewidth = 3)
    ax = plt.gca()
    ax.add_artist(gun)
    ax.add_artist(ball)
    ax.add_artist(sun)
    ax.add_artist(shadow)
    ax.set_xlim((0, 20))
    ax.set_ylim((-1, 21))
    plt.axis('off')
    fig.canvas.draw()
    imageset.append(torch.from_numpy(np.array(fig.canvas.renderer._renderer, dtype=np.float32)/255))
  plt.close('all')
  return torch.stack(imageset, dim=0) # bs x 4 x 96 x 96


def gen_pendulum_factors(random_state, batch_size):
  i_set = torch.tensor([random_state.randint(-40, 44) for _ in range(batch_size)])
  j_set = torch.tensor([random_state.randint(60, 148) for _ in range(batch_size)])
  shade_set = torch.tensor([random_state.randint(3, 13) for _ in range(batch_size)])
  mid_set = torch.tensor([random_state.randint(2, 20) for _ in range(batch_size)])
  
  return torch.stack([i_set, j_set, shade_set, mid_set], dim = 0).transpose(0,1) # 4 x batchsize