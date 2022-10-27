# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""Functions to create bandit problems from datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf


def one_hot(df, cols):
  """Returns one-hot encoding of DataFrame df including columns in cols."""
  for col in cols:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(col, axis=1)
  return df


def sample_mushroom_data(file_name,
                         num_contexts,
                         r_noeat=0,
                         r_eat_safe=5,
                         r_eat_poison_bad=-35,
                         r_eat_poison_good=5,
                         prob_poison_bad=0.5):
  """Samples bandit game from Mushroom UCI Dataset.

  Args:
    file_name: Route of file containing the original Mushroom UCI dataset.
    num_contexts: Number of points to sample, i.e. (context, action rewards).
    r_noeat: Reward for not eating a mushroom.
    r_eat_safe: Reward for eating a non-poisonous mushroom.
    r_eat_poison_bad: Reward for eating a poisonous mushroom if harmed.
    r_eat_poison_good: Reward for eating a poisonous mushroom if not harmed.
    prob_poison_bad: Probability of being harmed by eating a poisonous mushroom.

  Returns:
    dataset: Sampled matrix with n rows: (context, eat_reward, no_eat_reward).
    opt_vals: Vector of expected optimal (reward, action) for each context.

  We assume r_eat_safe > r_noeat, and r_eat_poison_good > r_eat_poison_bad.
  """

  # first two cols of df encode whether mushroom is edible or poisonous
  df = pd.read_csv(file_name, header=None)
  ind = np.random.choice(range(df.shape[0]), num_contexts, replace=True)

  # use attribute-"bruises" to generate specific groups: t-group and f-group
  tags = df.iloc[ind,4]
  tags = tags.tolist()

  df = one_hot(df, df.columns)

  contexts = df.iloc[ind, 2:]
  no_eat_reward = r_noeat * np.ones((num_contexts, 1))
  random_poison = np.random.choice(
      [r_eat_poison_bad, r_eat_poison_good],
      p=[prob_poison_bad, 1 - prob_poison_bad],
      size=num_contexts)
  eat_reward = r_eat_safe * df.iloc[ind, 0]
  eat_reward += np.multiply(random_poison, df.iloc[ind, 1])
  eat_reward = eat_reward.values.reshape((num_contexts, 1))

  # compute optimal expected reward and optimal actions
  exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad
  exp_eat_poison_reward += r_eat_poison_good * (1 - prob_poison_bad)
  opt_exp_reward = r_eat_safe * df.iloc[ind, 0] + max(
      r_noeat, exp_eat_poison_reward) * df.iloc[ind, 1]

  if r_noeat > exp_eat_poison_reward:
    # actions: no eat = 0 ; eat = 1
    opt_actions = df.iloc[ind, 0]  # indicator of edible
  else:
    # should always eat (higher expected reward)
    opt_actions = np.ones((num_contexts, 1))

  opt_vals = (opt_exp_reward.values, opt_actions.values)

  return np.hstack((contexts, no_eat_reward, eat_reward)), opt_vals, tags
