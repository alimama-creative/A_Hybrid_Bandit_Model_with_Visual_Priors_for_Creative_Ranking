from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pickle
from absl import app
from absl import flags
import numpy as np
import os
import logging
import tensorflow as tf

from HBM.data_sampler import sample_mushroom_data
from HBM.uniform_sampling import UniformSampling
from HBM.neural_linear_sampling import NeuralLinearPosteriorSampling
from HBM.contextual_bandit import run_contextual_bandit


FLAGS = flags.FLAGS
flags.DEFINE_string('datadir', 'data/mushroom.data', 'Directory where Mushroom data is stored')
flags.DEFINE_string('logdir', 'logs/', 'Directory where output will be saved')
flags.DEFINE_integer('context_dim', 117, 'Dimension of context information')
flags.DEFINE_integer('num_actions', 2, 'Numbers of actions to choose')
flags.DEFINE_integer('num_contexts', 50000, 'Numbers of contexts/samples')
flags.DEFINE_integer('exp_seed', 2, 'Seed for reproduce')

def print_configuration_op(FLAGS, logger):


    logger.info('My Configurations:')
    logger.info(' %s:\t %s'%('datadir', FLAGS.datadir))
    logger.info(' %s:\t %s'%('logdir', FLAGS.logdir))
    logger.info(' %s:\t %d'%('context_dim', FLAGS.context_dim))
    logger.info(' %s:\t %d'%('num_actions', FLAGS.num_actions))
    logger.info(' %s:\t %d'%('num_contexts', FLAGS.num_contexts))
    logger.info('End of configuration')

def display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, logger):
    """Displays summary statistics of the performance of each algorithm."""
    logger.info('---------------------------------------------------')
    logger.info('Bandit completed after {} seconds.'.format(time.time() - t_init))
    logger.info('---------------------------------------------------')

    performance_pairs = []
    for j, a in enumerate(algos):
        performance_pairs.append((a.name, np.sum(h_rewards[:, j])))
    performance_pairs = sorted(performance_pairs,
                                key=lambda elt: elt[0],
                                reverse=False)
    for i, (name, reward) in enumerate(performance_pairs):
        logger.info('{:3}) {:20}| \t \t total reward = {:10}.'.format(i, name, reward))


    norm_regret = (np.sum(opt_rewards) - performance_pairs[1][1])/(np.sum(opt_rewards) - performance_pairs[0][1])
    logger.info('---------------------------------------------------')
    logger.info('Optimal total reward = {}.'.format(np.sum(opt_rewards)))
    logger.info('Normalized Regret = {}.'.format(norm_regret))
    logger.info('---------------------------------------------------')


def main(_):

    datadir = FLAGS.datadir
    logdir = FLAGS.logdir
    context_dim = FLAGS.context_dim
    num_contexts = FLAGS.num_contexts
    num_actions = FLAGS.num_actions
    tf.set_random_seed(0)
    #np.random.seed(FLAGS.exp_seed)
    #import random
    #random.seed(FLAGS.exp_seed)
    #get TF logger
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S")
    fh = logging.FileHandler(FLAGS.logdir + timestamp +'-log.txt')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    print_configuration_op(FLAGS, logger)


    dataset, opt_mushroom, tags = sample_mushroom_data(datadir, num_contexts)
    opt_rewards, opt_actions = opt_mushroom

    # Define hyperparameters and algorithms
    hparams = tf.contrib.training.HParams(num_actions=num_actions)

    #The following hyperparameters are set as the original implementation in "DEEP BAYESIAN BANDITS SHOWDOWN"
    hparams_hbm = tf.contrib.training.HParams(
        algo_type="hybrid_bandit_model",
        num_actions=num_actions,
        context_dim=context_dim,
        init_scale=0.3,
        activation=tf.nn.relu,
        layer_sizes=[50],
        batch_size=512,
        activate_decay=True,
        initial_lr=0.1,
        max_grad_norm=5.0,
        show_training=False,
        freq_summary=1000,
        buffer_s=-1,
        initial_pulls=2,
        reset_lr=True,
        lr_decay_rate=0.5,
        training_freq=10,
        training_freq_network=50,
        training_epochs=100,
        a0=6.0,
        b0=6.0,
        lambda_prior=0.25)

    feature_set = set(tags)

    algos = [
        UniformSampling('Uniform_Sampling', hparams),
        NeuralLinearPosteriorSampling('hybrid_bandit_model', hparams_hbm, feature_set)
    ]

    # Run contextual bandit problem
    t_init = time.time()
    results = run_contextual_bandit(context_dim, num_actions, dataset, tags, algos, FLAGS)
    _, h_rewards = results

    # Display results
    display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, logger)

if __name__ == '__main__':
  app.run(main)
