
"""Implementation of the disentanglement metric from the FactorVAE paper.

Based on "Disentangling by Factorising" (https://arxiv.org/abs/1802.05983).
"""
from .m_utils import obtain_representation, gen_pendulum_factors, gen_pendulum_observation
import numpy as np
import torch
device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")


def compute_factor_vae(ground_truth_data,
                       representation_function,
                       random_state,
                       batch_size,
                       num_train,
                       num_eval,
                       num_variance_estimate):
  """Computes the FactorVAE disentanglement metric.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    batch_size: Number of points to be used to compute the training_sample.
    num_train: Number of points used for training.
    num_eval: Number of points used for evaluation.
    num_variance_estimate: Number of points used to estimate global variances.

  Returns:
    Dictionary with scores:
      train_accuracy: Accuracy on training set.
      eval_accuracy: Accuracy on evaluation set.
  """
  global_variances = _compute_variances(ground_truth_data,representation_function,num_variance_estimate, random_state)
  active_dims = _prune_dims(global_variances)
  scores_dict = {}

  if not active_dims.any():
    scores_dict["train_accuracy"] = 0.
    scores_dict["eval_accuracy"] = 0.
    scores_dict["num_active_dims"] = 0
    return scores_dict

  training_votes = _generate_training_batch(ground_truth_data, representation_function, batch_size,num_train, random_state,global_variances, active_dims)
  classifier = np.argmax(training_votes, axis=0)
  other_index = np.arange(training_votes.shape[1])

  print("Evaluate training set accuracy.")
  train_accuracy = np.sum(training_votes[classifier, other_index]) * 1. / np.sum(training_votes)
  print("Training set accuracy: %.2g", train_accuracy)

  print("Generating evaluation set.")
  eval_votes = _generate_training_batch(ground_truth_data,representation_function, batch_size,num_eval, random_state,global_variances, active_dims)

  print("Evaluate evaluation set accuracy.")
  eval_accuracy = np.sum(eval_votes[classifier,other_index]) * 1. / np.sum(eval_votes)
  print("Evaluation set accuracy: %.2g", eval_accuracy)
  scores_dict["train_accuracy"] = train_accuracy
  scores_dict["eval_accuracy"] = eval_accuracy
  scores_dict["num_active_dims"] = len(active_dims)
  return scores_dict

def _prune_dims(variances, threshold=0.):
  """Mask for dimensions collapsed to the prior."""
  scale_z = np.sqrt(variances)
  return scale_z >= threshold


def _compute_variances(ground_truth_data,representation_function,batch_size,random_state,eval_batch_size=64):
  """Computes the variance for each dimension of the representation.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the variances.
    random_state: Numpy random state used for randomness.
    eval_batch_size: Batch size used to eval representation.

  Returns:
    Vector with the variance of each dimension.
  """
  num_factors=4
  observations = gen_pendulum_observation(gen_pendulum_factors(random_state, batch_size))
  representations = obtain_representation(observations,representation_function,eval_batch_size)
  representations = np.transpose(representations) # batch_size x num_factors
  assert representations.shape[0] == batch_size
  return np.var(representations, axis=0, ddof=1)


def _generate_training_sample(ground_truth_data, representation_function, batch_size, random_state, global_variances,active_dims):
  """Sample a single training sample based on a mini-batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the training_sample.
    random_state: Numpy random state used for randomness.
    global_variances: Numpy vector with variances for all dimensions of
      representation.
    active_dims: Indexes of active dimensions.

  Returns:
    factor_index: Index of factor coordinate to be used.
    argmin: Index of representation coordinate with the least variance.
  """
  num_factors=4
  # Select random coordinate to keep fixed.
  factor_index = random_state.randint(num_factors)
  # Sample two mini batches of latent variables.
  factors =  gen_pendulum_factors(random_state, batch_size)
  # Fix the selected factor across mini-batch.
  factors[:, factor_index] = factors[0, factor_index]
  # Obtain the observations.
  observations = gen_pendulum_observation(factors)
  representations = representation_function(observations.to(device)).detach().cpu().numpy()
  local_variances = np.var(representations, axis=0, ddof=1)
  argmin = np.argmin(local_variances[active_dims] /global_variances[active_dims])
  return factor_index, argmin


def _generate_training_batch(ground_truth_data, representation_function,batch_size, num_points, random_state,global_variances, active_dims):
  """Sample a set of training samples based on a batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    batch_size: Number of points to be used to compute the training_sample.
    num_points: Number of points to be sampled for training set.
    random_state: Numpy random state used for randomness.
    global_variances: Numpy vector with variances for all dimensions of
      representation.
    active_dims: Indexes of active dimensions.

  Returns:
    (num_factors, dim_representation)-sized numpy array with votes.
  """
  num_factors = 4
  votes = np.zeros((num_factors, global_variances.shape[0]), dtype=np.int64)
  for _ in range(num_points):
    factor_index, argmin = _generate_training_sample(ground_truth_data,representation_function,batch_size, random_state,global_variances,active_dims)
    votes[factor_index, argmin] += 1
  return votes


if __name__ == "__main__":
    ground_truth_data = 1
    representation_function = lambda x: x
    random_state = np.random.RandomState(0)
    scores = compute_factor_vae(ground_truth_data, representation_function, random_state, None, 5, 3000,2000, 2500)
