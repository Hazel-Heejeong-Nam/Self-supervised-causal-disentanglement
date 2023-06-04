
"""Implementation of the disentanglement metric from the BetaVAE paper.

Based on "beta-VAE: Learning Basic Visual Concepts with a Constrained
Variational Framework" (https://openreview.net/forum?id=Sy2fzU9gl).
"""

import numpy as np
from sklearn import linear_model
from .m_utils import gen_pendulum_observation, gen_pendulum_factors
import torch
from tqdm import trange
device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")


def compute_beta_vae_sklearn(ground_truth_data,representation_function,random_state,batch_size,num_train,num_eval):

  train_points, train_labels = _generate_training_batch(ground_truth_data, representation_function, batch_size, num_train,random_state)

  model = linear_model.LogisticRegression(random_state=random_state)
  model.fit(train_points, train_labels)

  print("Evaluate training set accuracy.")
  train_accuracy = model.score(train_points, train_labels)
  train_accuracy = np.mean(model.predict(train_points) == train_labels)
  print(f"Training set accuracy:{train_accuracy}")

  print("Generating evaluation set.")
  eval_points, eval_labels = _generate_training_batch(ground_truth_data, representation_function, batch_size, num_eval,random_state)

  print("Evaluate evaluation set accuracy.")
  eval_accuracy = model.score(eval_points, eval_labels)
  print(f"Evaluation set accuracy: {eval_accuracy}")
  scores_dict = {}
  scores_dict["train_accuracy"] = train_accuracy
  scores_dict["eval_accuracy"] = eval_accuracy
  return scores_dict


def _generate_training_batch(ground_truth_data, representation_function,batch_size, num_points, random_state):

  points = None  # Dimensionality depends on the representation function.
  labels = np.zeros(num_points, dtype=np.int64)
  for i in trange(num_points):
    labels[i], feature_vector = _generate_training_sample(ground_truth_data, representation_function, batch_size, random_state)
    if points is None:
      points = np.zeros((num_points, feature_vector.shape[0]))
    points[i, :] = feature_vector
  return points, labels


def _generate_training_sample(ground_truth_data, representation_function,batch_size, random_state):

  num_factors=4 # 일단 하드코딩
  # Select random coordinate to keep fixed.
  index = random_state.randint(num_factors)
  # Sample two mini batches of latent variables.

  factors1 = gen_pendulum_factors(random_state, batch_size) # bs x numfactors
  factors2 = gen_pendulum_factors(random_state, batch_size)
  # Ensure sampled coordinate is the same across pairs of samples.
  factors2[:, index] = factors1[:, index]
  # Transform latent variables to observation space.
  observation1 = gen_pendulum_observation(factors1)
  observation2 = gen_pendulum_observation(factors2)
  # Compute representations based on the observations.
  
  with torch.no_grad():
    representation1 = representation_function(observation1.to(device))
    representation2 = representation_function(observation2.to(device))
  # Compute the feature vector based on differences in representation.
  feature_vector = torch.mean(torch.abs(representation1- representation2), dim=0)
  return index, feature_vector.cpu().numpy()

if __name__ == "__main__":
    
    ground_truth_data = 1
    representation_function = lambda x: x
    random_state = np.random.RandomState(0)
    scores = compute_beta_vae_sklearn(ground_truth_data, representation_function, random_state, 5,2000, 2000)
