#built-in package
import os
import math
import time
import random
import datetime
import argparse
from functools import reduce
from collections import defaultdict
from multiprocessing import Process,Pool

#third package
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma

class creative_info(object):
    def __init__(self):
        self.pv = 0
        self.clk = 0
    def add(self, pv, clk):
        self.pv = pv
        self.clk = clk

def sigmoid(x):
    x = (x - 150)/50
    return 1/(1+np.exp(-x))

def read_file(path_to_file):
    """
        The keys' order of items is (item_id, ds, image_name)

        Args:
            path_to_file: The path to file you want to read

        Return:
            items based on key-value
    """
    items = defaultdict(lambda :defaultdict(lambda : defaultdict(creative_info)))
    feats = defaultdict(lambda: defaultdict(list))

    with open(path_to_file, mode='r') as f:
        lines = f.readlines()
        for line in lines:
            item_id, image, ds, pv, clk, feat = line.strip().split(' ')
            feat = [float(x)/100.0 for x in feat.split(',')]
            items[item_id][int(ds)][image].add(int(pv), int(clk))
            feats[item_id][image] = feat
    return items, feats

#bandit algorithms
def random_(item_id_list):
    """
        Random strategy:select one image from item's image set randomly.

        Args:
            item_id_list:all items in the test dataset.

        Return:
            num_clk: total click in random strategy.
            num_pv: total pv in random strategy.
    """
    num_clk = [0.0] * num_days
    num_pv = [0.0] * num_days

    for item_id in item_id_list:
        cur_item = items[item_id]
        ds_list = list(cur_item.keys())
        img_list = list(cur_item[ds_list[0]].keys())

        for idx, ds in enumerate(ds_list):
            cur_item_ds = cur_item[ds]
            clk_data = []
            for img in img_list:
                clk_data.extend([(img, 0)]*(cur_item_ds[img].pv - cur_item_ds[img].clk))
                clk_data.extend([(img, 1)]*cur_item_ds[img].clk)
            random.shuffle(clk_data)

            select_list = list(range(len(img_list)))
            for sample in clk_data:
                random_index = random.choice(select_list)
                if img_list[random_index] == sample[0]:
                    num_clk[idx] += sample[1]
                    num_pv[idx] += 1

    return num_clk, num_pv

def e_greedy(item_id_list):
    """
        E_Greedy strategy:select image based on e-value from item's image set.
        Daily update:accumulate one day's click to update parameters.
        Args:
            item_id_list:all items in the test dataset.

        Return:
            num_clk: total click in E_Greedy strategy.
            num_pv: total pv in E_Greedy strategy.
    """
    epsilon = 1e-5 #for numerical stability
    num_pv = [0] * num_days
    num_clk = [0] * num_days

    for item_id in item_id_list:

        cur_item = items[item_id]

        ds_list = sorted(cur_item.keys())
        image_list = sorted(list(cur_item[ds_list[0]].keys()))
        select_list = list(range(len(image_list)))

        image_num_pv = np.zeros((len(image_list), len(ds_list)))+epsilon
        image_num_clk = np.zeros((len(image_list), len(ds_list)))

        for i,ds in enumerate(ds_list):
            cur_item_ds = cur_item[ds]

            clk_data = []
            for img in image_list:
                clk_data.extend([(img, 0)]*(cur_item_ds[img].pv - cur_item_ds[img].clk))
                clk_data.extend([(img, 1)]*cur_item_ds[img].clk)
            random.shuffle(clk_data)

            if i == 0:
                for sample in clk_data:
                    pick = random.choice(select_list)
                    if image_list[pick] == sample[0]:
                        num_pv[i] += 1
                        num_clk[i] += sample[1]
                        image_num_pv[pick][i]+= 1
                        image_num_clk[pick][i]+=sample[1]
                continue

            ctr = np.array([np.sum(image_num_clk[k,:i])/np.sum(image_num_pv[k,:i]) for k in range(len(image_list))])
            index = np.argmax(ctr)

            for sample in clk_data:
                if random.random()<e:
                    pick = random.choice(select_list)
                    if image_list[pick] == sample[0]:
                        num_pv[i] += 1
                        num_clk[i] += sample[1]
                        image_num_pv[pick][i]+= 1
                        image_num_clk[pick][i]+=sample[1]
                else:
                    if image_list[index] == sample[0]:
                        num_pv[i] += 1
                        num_clk[i] += sample[1]
                        image_num_pv[index][i]+= 1
                        image_num_clk[index][i]+=sample[1]

    return num_clk, num_pv

def thompson(item_id_list):
    """
        Thompson Sampling strategy:select image based on posteriori distribution.

        Args:
            item_id_list:all items in the test dataset.

        Return:
            num_clk: total click in Thompson Sampling strategy.
            num_pv: total pv in Thompson Sampling strategy.
    """
    num_pv = [0] * num_days
    num_clk = [0] * num_days

    for item_id in item_id_list:
        cur_item = items[item_id]

        ds_list = sorted(cur_item.keys())
        image_list = sorted(list(cur_item[ds_list[0]].keys()))
        select_list = list(range(len(image_list)))

        image_num_pv = np.zeros((len(image_list), len(ds_list)))
        image_num_clk = np.zeros((len(image_list), len(ds_list)))

        for i,ds in enumerate(ds_list):
            cur_item_ds = cur_item[ds]

            clk_data = []
            for img in image_list:
                clk_data.extend([(img, 0)]*(cur_item_ds[img].pv - cur_item_ds[img].clk))
                clk_data.extend([(img, 1)]*cur_item_ds[img].clk)
            random.shuffle(clk_data)

            if i == 0:
                for sample in clk_data:
                    pick = random.choice(select_list)
                    if image_list[pick] == sample[0]:
                        num_pv[i] += 1
                        num_clk[i] += sample[1]
                        image_num_pv[pick][i]+= 1
                        image_num_clk[pick][i]+=sample[1]
                continue

            ctr = np.array([np.random.beta(sum(image_num_clk[k][:i])+3.43046, sum(image_num_pv[k][:i])-sum(image_num_clk[k][:i])+109.78302, len(clk_data))
                            for k in range(len(image_list))])

            for j, sample in enumerate(clk_data):
                index = np.argmax(ctr[:, j])
                if image_list[index] == sample[0]:
                    num_pv[i] += 1
                    num_clk[i] += sample[1]
                    image_num_pv[index][i]+= 1
                    image_num_clk[index][i]+=sample[1]

    return num_clk, num_pv

def neural_thompson(item_id_list):
    """
        Neural Thompson strategy:select image based on posteriori distribution.
        warning:mu is set to (0,)*10 which means no listnet's prior
        Args:
            item_id_list:all items in the test dataset.

        Return:
            num_clk: total click in Neural Thompson strategy.
            num_pv: total pv in Neural Thompson strategy.
    """
    a0 = 60000.
    b0 = 6.
    mu0 = np.zeros((10,1))
    mu0 = np.array([0.1]*10).reshape((10, 1))
    cov0 = 4.0 * np.eye(10)

    num_pv = [0] * num_days
    num_clk = [0] * num_days

    for item_id in item_id_list:
        cur_item = items[item_id]

        ds_list = sorted(cur_item.keys())
        image_list = sorted(list(cur_item[ds_list[0]].keys()))

        image_num_pv = np.zeros((len(image_list), len(ds_list)))
        image_num_clk = np.zeros((len(image_list), len(ds_list)))

        for i,ds in enumerate(ds_list):
            cur_item_ds = cur_item[ds]

            clk_data = []
            for img in image_list:
                clk_data.extend([(img, 0)] * (cur_item_ds[img].pv - cur_item_ds[img].clk))
                clk_data.extend([(img, 1)] * cur_item_ds[img].clk)
            random.shuffle(clk_data)

            a = a0
            b = b0
            mu = mu0
            cov = cov0

            if i>0:
                z = [np.array(feats[item_id][image]).reshape((1, 10)) for image in image_list]
                img_pv = [sum(image_num_pv[j, :i]) for j in range(len(image_list))]
                img_clk = [sum(image_num_clk[j, :i]) for j in range(len(image_list))]

                total_clk = sum(sum(image_num_clk[:,:i]))
                total_pv = sum(sum(image_num_pv[:,:i]))

                if total_pv > 0:
                    s = [np.dot(z[k].T, z[k]) * img_pv[k] for k in range(len(image_list))]
                    sum_s = reduce(lambda x,y:x+y, s)
                    precision_a = sum_s + 0.25 * np.eye(10)
                    cov_a = np.linalg.inv(precision_a)
                    s = np.dot(0.25*np.eye(10), mu0)
                    tmp = [z[k].T*img_clk[k] for k in range(len(image_list))]
                    sum_tmp = reduce(lambda x,y: x+y, tmp)

                    mu_a = np.dot(cov_a, (s + sum_tmp))

                    a_post = a0 + total_pv/2.
                    b_upd = 0.5 * total_clk

                    b_upd = b_upd + 0.5 * (np.dot(mu0.T, np.dot(cov0, mu0)) - np.dot(mu_a.T, np.dot(precision_a, mu_a)))

                    b_post = b0 + b_upd
                    mu = mu_a
                    cov = cov_a
                    precision = precision_a
                    a = a_post
                    b = b_post.reshape(1)[0]

            sigma2_s = b * invgamma.rvs(a, size=len(clk_data))

            for j in range(len(clk_data)):
                beta_s = np.random.multivariate_normal(np.array(mu).reshape(-1), sigma2_s[j].reshape(-1) * cov)

                vals = [
                    np.dot(beta_s, np.array(feats[item_id][image_list[k]]).reshape((1,10)).T)
                    for k in range(len(image_list))]

                vals = np.array(vals).reshape(len(image_list))
                index = np.argmax(vals)

                if image_list[index] == clk_data[j][0]:
                    num_pv[i] += 1
                    num_clk[i] += clk_data[j][1]
                    image_num_pv[index][i]+= 1
                    image_num_clk[index][i]+=clk_data[j][1]

    return num_clk, num_pv

def ucb(item_id_list):
    """
        UCB strategy:select image based on ucb value

        Args:
            item_id_list:all items in the test dataset.

        Return:
            num_clk: total click in UCB strategy.
            num_pv: total pv in UCB strategy.
    """
    class tmp_info(object):
        def __init__(self):
            self.mean_reward = 0
            self.n_t = 1e-5
        def add(self, reward):
            self.mean_reward = ((self.mean_reward*self.n_t)+reward)/(self.n_t+1)
            self.n_t += 1

    num_pv = [0] * num_days
    num_clk = [0] * num_days

    for item_id in item_id_list:
        cur_item = items[item_id]

        Q = defaultdict(tmp_info)

        total_pv = 0
        ds_list = sorted(cur_item.keys())
        image_list = sorted(list(cur_item[ds_list[0]].keys()))
        select_list = list(range(len(image_list)))

        image_num_pv = np.zeros((len(image_list), len(ds_list)))
        image_num_clk = np.zeros((len(image_list), len(ds_list)))

        for i,ds in enumerate(ds_list):
            cur_item_ds = cur_item[ds]

            clk_data = []
            for img in image_list:
                clk_data.extend([(img, 0)] * (cur_item_ds[img].pv - cur_item_ds[img].clk))
                clk_data.extend([(img, 1)] * cur_item_ds[img].clk)
            random.shuffle(clk_data)

            if i == 0 or total_pv == 0:
                for sample in clk_data:
                    pick = random.choice(select_list)
                    if image_list[pick] == sample[0]:
                        num_pv[i] += 1
                        num_clk[i] += sample[1]
                        image_num_clk[pick][i] += sample[1]
                        image_num_pv[pick][i] += 1
                        Q[image_list[pick]].add(sample[1])
                        total_pv += 1
                continue

            ucb_value = np.array([Q[img].mean_reward+lamda*math.sqrt(2*math.log(total_pv)/Q[img].n_t) for img in image_list])
            index = np.argmax(ucb_value)

            for sample in clk_data:
                if image_list[index] == sample[0]:
                    num_pv[i] += 1
                    num_clk[i] += sample[1]
                    image_num_clk[index][i] += sample[1]
                    image_num_pv[index][i] += 1
                    Q[image_list[index]].add(sample[1])
                    total_pv += 1

    return num_clk, num_pv

def neural_ucb(item_id_list):
    """
        Neural UCB strategy:select image based on Neural UCB value

        warning1: default setting does not contain listnet's prior,if you
        want to set prior,please make b0,theta0 to be 0.1

        warning2: this is itemwise neural ucb

        Args:
            item_id_list:all items in the test dataset.

        Return:
            num_clk: total click in Neural UCB strategy.
            num_pv: total pv in Neural UCB strategy.
    """

    A0 = np.eye(10)
    A0_inv = np.eye(10)
    # b0 = np.ones((10,1))*0.1
    # theta0 = np.ones((10,1))*0.1
    b0 = np.zeros((10,1))
    theta0 = np.zeros((10,1))

    num_pv = [0] * num_days
    num_clk = [0] * num_days

    for item_id in item_id_list:
        cur_item = items[item_id]

        ds_list = sorted(cur_item.keys())
        image_list = sorted(list(cur_item[ds_list[0]].keys()))

        image_num_pv = np.zeros((len(image_list), len(ds_list)))
        image_num_clk = np.zeros((len(image_list), len(ds_list)))

        for i,ds in enumerate(ds_list):
            cur_item_ds = cur_item[ds]

            clk_data = []
            for img in image_list:
                clk_data.extend([(img, 0)] * (cur_item_ds[img].pv - cur_item_ds[img].clk))
                clk_data.extend([(img, 1)] * cur_item_ds[img].clk)
            random.shuffle(clk_data)

            A = A0
            A_inv = A0_inv
            b = b0
            theta = theta0

            if i>0:
                z = [np.array(feats[item_id][image]).reshape((1, 10)) for image in image_list]
                img_pv = [sum(image_num_pv[j, :i]) for j in range(len(image_list))]
                img_clk = [sum(image_num_clk[j, :i]) for j in range(len(image_list))]

                total_pv = sum(sum(image_num_pv[:,:i]))

                if total_pv > 0:
                    s = [np.dot(z[k].T, z[k]) * img_pv[k] for k in range(len(image_list))]
                    sum_s = reduce(lambda x,y:x+y, s)
                    A = A + sum_s
                    A_inv = np.linalg.inv(A)

                    tmp = [z[k].T*img_clk[k] for k in range(len(image_list))]
                    sum_tmp = reduce(lambda x,y: x+y, tmp)
                    b = b + sum_tmp

                    theta = np.matmul(A_inv, b)

            z = [np.array(feats[item_id][image]).reshape((1, 10)) for image in image_list]
            for j in range(len(clk_data)):
                p = [np.dot(theta.reshape(-1), z[k].reshape(-1)).item() \
                    + lamda*np.sqrt(np.matmul(z[k], np.matmul(A_inv, z[k].reshape((10,1))))).item() \
                    for k in range(len(image_list))]

                index = np.argmax(p)

                if image_list[index] == clk_data[j][0]:
                    num_pv[i] += 1
                    num_clk[i] += clk_data[j][1]
                    image_num_pv[index][i]+= 1
                    image_num_clk[index][i]+=clk_data[j][1]

    return num_clk, num_pv

def neural_linear_item_wise(item_id_list):
    """
        Neural linear(itemwise) strategy:select image based on postprior distibution
        Args:
            item_id_list:all items in the test dataset.

        Return:
            num_clk: total click in Neural Linear(itemwise) strategy.
            num_pv: total pv in Neural Linear(itemwise) strategy.
    """
    a0 = 60000.
    b0 = 6.
    mu0 = (np.ones(10)*0.1).reshape(-1, 1)
    cov0 = 4.0 * np.eye(10)

    num_pv = [0] * num_days
    num_clk = [0] * num_days

    for item_id in item_id_list:
        cur_item = items[item_id]

        ds_list = sorted(cur_item.keys())
        image_list = sorted(list(cur_item[ds_list[0]].keys()))

        image_num_pv = np.zeros((len(image_list), len(ds_list)))
        image_num_clk = np.zeros((len(image_list), len(ds_list)))

        for i,ds in enumerate(ds_list):
            cur_item_ds = cur_item[ds]

            clk_data = []
            for img in image_list:
                clk_data.extend([(img, 0)] * (cur_item_ds[img].pv - cur_item_ds[img].clk))
                clk_data.extend([(img, 1)] * cur_item_ds[img].clk)
            random.shuffle(clk_data)

            a = a0
            b = b0
            mu = mu0
            cov = cov0

            if i>0:
                z = [np.array(feats[item_id][image]).reshape((1, 10)) for image in image_list]
                img_pv = [sum(image_num_pv[j, :i]) for j in range(len(image_list))]
                img_clk = [sum(image_num_clk[j, :i]) for j in range(len(image_list))]

                total_clk = sum(sum(image_num_clk[:,:i]))
                total_pv = sum(sum(image_num_pv[:,:i]))

                if total_pv > 0:
                    s = [np.dot(z[k].T, z[k]) * img_pv[k] for k in range(len(image_list))]
                    sum_s = reduce(lambda x,y:x+y, s)
                    precision_a = sum_s + 0.25 * np.eye(10)
                    cov_a = np.linalg.inv(precision_a)
                    s = np.dot(0.25*np.eye(10), mu0)
                    tmp = [z[k].T*img_clk[k] for k in range(len(image_list))]
                    sum_tmp = reduce(lambda x,y: x+y, tmp)

                    mu_a = np.dot(cov_a, (s + sum_tmp))

                    a_post = a0 + total_pv/2.
                    b_upd = 0.5 * total_clk

                    b_upd = b_upd + 0.5 * (np.dot(mu0.T, np.dot(cov0, mu0)) - np.dot(mu_a.T, np.dot(precision_a, mu_a)))

                    b_post = b0 + b_upd
                    mu = mu_a
                    cov = cov_a
                    precision = precision_a
                    a = a_post
                    b = b_post.reshape(1)[0]

            sigma2_s = b * invgamma.rvs(a, size=len(clk_data))

            for j in range(len(clk_data)):
                beta_s = np.random.multivariate_normal(np.array(mu).reshape(-1), sigma2_s[j].reshape(-1) * cov)

                vals = [
                    np.dot(beta_s, np.array(feats[item_id][image_list[k]]).reshape((1,10)).T)
                    for k in range(len(image_list))]

                vals = np.array(vals).reshape(len(image_list))
                index = np.argmax(vals)

                if image_list[index] == clk_data[j][0]:
                    num_pv[i] += 1
                    num_clk[i] += clk_data[j][1]
                    image_num_pv[index][i]+= 1
                    image_num_clk[index][i]+=clk_data[j][1]

    return num_clk, num_pv

def neural_linear_creative_wise(item_id_list):
    """
        Neural linear(creativewise) strategy:select image based on postprior distibution
        Args:
            item_id_list:all items in the test dataset.

        Return:
            num_clk: total click in Neural Linear(creativewise) strategy.
            num_pv: total pv in Neural Linear(creativewise) strategy.
    """
    a0 = 60000.
    b0 = 6.
    mu0 = (np.ones(10)*0.1).reshape(-1, 1)
    cov0 = 4.0 * np.eye(10)
    precision0 = 0.25 * np.eye(10)

    num_pv = [0] * num_days
    num_clk = [0] * num_days

    for item_id in item_id_list:
        cur_item = items[item_id]

        ds_list = sorted(cur_item.keys())
        image_list = sorted(list(cur_item[ds_list[0]].keys()))

        image_num_pv = np.zeros((len(image_list), len(ds_list)))
        image_num_clk = np.zeros((len(image_list), len(ds_list)))

        for i,ds in enumerate(ds_list):
            cur_item_ds = cur_item[ds]

            clk_data = []
            for img in image_list:
                clk_data.extend([(img, 0)] * (cur_item_ds[img].pv - cur_item_ds[img].clk))
                clk_data.extend([(img, 1)] * cur_item_ds[img].clk)
            random.shuffle(clk_data)

            mu = [mu0.reshape(-1) for _ in range(len(image_list))]
            cov = [cov0 for _ in range(len(image_list))]
            a = [a0]*len(image_list)
            b = [b0]*len(image_list)
            precision = [precision0 for _ in range(len(image_list))]

            if i>0:
                for j in range(len(image_list)):
                    image = image_list[j]
                    z = np.array(feats[item_id][image]).reshape((1, 10))
                    pv = sum(image_num_pv[j, :i])
                    clk = sum(image_num_clk[j, :i])
                    if pv > 0:
                        s = np.dot(z.T, z) * pv
                        precision_a = s + 0.25 * np.eye(10)
                        cov_a = np.linalg.inv(precision_a)
                        s = np.dot(0.25*np.eye(10), mu0)
                        mu_a = np.dot(cov_a, (s+z.T * clk))
                        a_post = a0 + pv/2.
                        b_upd = 0.5 * clk
                        b_upd = b_upd + 0.5 * (np.dot(mu0.T, np.dot(cov0, mu0)) - np.dot(mu_a.T, np.dot(precision_a, mu_a)))
                        b_post = b0 + b_upd
                        mu[j] = mu_a
                        cov[j] = cov_a
                        precision[j] = precision_a
                        a[j] = a_post
                        b[j] = b_post.reshape(1)[0]

            sigma2_s = [b[k] * invgamma.rvs(a[k], size=len(clk_data)) for k in range(len(image_list))]
            for j in range(len(clk_data)):
                beta_s = [np.random.multivariate_normal(np.array(mu[k]).reshape(-1), sigma2_s[k][j].reshape(-1) * cov[k])
                            for k in range(len(image_list))]

                vals = [
                    np.dot(beta_s[k], np.array(feats[item_id][image_list[k]]).reshape((1,10)).T)
                    for k in range(len(image_list))]

                vals = np.array(vals).reshape(len(image_list))
                index = np.argmax(vals)

                if image_list[index] == clk_data[j][0]:
                    num_pv[i] += 1
                    num_clk[i] += clk_data[j][1]
                    image_num_pv[index][i]+= 1
                    image_num_clk[index][i]+=clk_data[j][1]

    return num_clk, num_pv

def hybrid_model(item_id_list):
    """
        Neural linear(hybrid) strategy:select image based on postprior distibution
        alpha = sigmoid(item_pv)
        score = (1-alpha)*item_score +  alpha*creative_score
        Args:
            item_id_list:all items in the test dataset.

        Return:
            num_clk: total click in Neural Linear(hybrid) strategy.
            num_pv: total pv in Neural Linear(hybrid) strategy.
    """
    a0 = 60000.
    b0 = 6.
    mu0 = (np.ones(10)*0.1).reshape(-1, 1)
    cov0 = 4.0 * np.eye(10)
    precision0 = 0.25 * np.eye(10)

    num_pv = [0] * num_days
    num_clk = [0] * num_days

    for item_id in item_id_list:
        cur_item = items[item_id]

        ds_list = sorted(cur_item.keys())
        image_list = sorted(list(cur_item[ds_list[0]].keys()))

        image_num_pv = np.zeros((len(image_list), len(ds_list)))
        image_num_clk = np.zeros((len(image_list), len(ds_list)))

        for i,ds in enumerate(ds_list):
            cur_item_ds = cur_item[ds]

            clk_data = []
            for img in image_list:
                clk_data.extend([(img, 0)] * (cur_item_ds[img].pv - cur_item_ds[img].clk))
                clk_data.extend([(img, 1)] * cur_item_ds[img].clk)
            random.shuffle(clk_data)

            #item-wise W
            a_item = a0
            b_item = b0
            mu_item = mu0
            cov_item = cov0
            precision_item = precision0

            #creative-wise W
            a_c = [a0 for _ in range(len(image_list))]
            b_c = [b0 for _ in range(len(image_list))]
            mu_c = [mu0 for _ in range(len(image_list))]
            cov_c = [cov0 for _ in range(len(image_list))]
            precision_c = [precision0 for _ in range(len(image_list))]

            if i>0:
                #item-wise W's update
                z = [np.array(feats[item_id][image]).reshape((1, 10)) for image in image_list]
                img_pv = [sum(image_num_pv[j, :i]) for j in range(len(image_list))]
                img_clk = [sum(image_num_clk[j, :i]) for j in range(len(image_list))]

                total_clk = sum(sum(image_num_clk[:,:i]))
                total_pv = sum(sum(image_num_pv[:,:i]))

                if total_pv > 0:
                    s = [np.dot(z[k].T, z[k]) * img_pv[k] for k in range(len(image_list))]
                    sum_s = reduce(lambda x,y:x+y, s)
                    precision_a = sum_s + precision0
                    cov_a = np.linalg.inv(precision_a)
                    s = np.dot(precision0, mu0)
                    tmp = [z[k].T*img_clk[k] for k in range(len(image_list))]
                    sum_tmp = reduce(lambda x,y: x+y, tmp)

                    mu_a = np.dot(cov_a, (s + sum_tmp))

                    a_post = a0 + total_pv/2.
                    b_upd = 0.5 * total_clk

                    b_upd = b_upd + 0.5 * (np.dot(mu0.T, np.dot(cov0, mu0)) - np.dot(mu_a.T, np.dot(precision_a, mu_a)))

                    b_post = b0 + b_upd
                    mu_item = mu_a
                    cov_item = cov_a
                    precision_item = precision_a
                    a_item = a_post
                    b_item = b_post.reshape(1)[0]

                for t in range(len(image_list)):
                    image = image_list[t]
                    z = np.array(feats[item_id][image]).reshape((1, 10))
                    pv = sum(image_num_pv[t, :i])
                    clk = sum(image_num_clk[t, :i])
                    if pv > 0:
                        s = np.dot(z.T, z) * pv
                        precision_a = s + precision0
                        cov_a = np.linalg.inv(precision_a)
                        s = np.dot(precision0, mu0)
                        mu_a = np.dot(cov_a, (s+z.T * clk))
                        a_post = a0 + pv/2.
                        b_upd = 0.5 * clk
                        b_upd = b_upd + 0.5 * (np.dot(mu0.T, np.dot(cov0, mu0)) - np.dot(mu_a.T, np.dot(precision_a, mu_a)))
                        b_post = b0 + b_upd
                        mu_c[t] = mu_a
                        cov_c[t] = cov_a
                        precision_c[t] = precision_a
                        a_c[t] = a_post
                        b_c[t] = b_post.reshape(1)[0]

            #item-wise creative sigma
            sigma2_s_item = b_item * invgamma.rvs(a_item, size=len(clk_data))
            #creative-wise creative sigma
            sigma2_s_c = [b_c[k] * invgamma.rvs(a_c[k], size=len(clk_data)) for k in range(len(image_list))]

            for j in range(len(clk_data)):
                #item-wise w
                beta_s_item = np.random.multivariate_normal(np.array(mu_item).reshape(-1), sigma2_s_item[j].reshape(-1) * cov_item)
                #creative-wise w
                beta_s_c = [np.random.multivariate_normal(np.array(mu_c[k]).reshape(-1), sigma2_s_c[k][j].reshape(-1) * cov_c[k])
                            for k in range(len(image_list))]

                #item-wise value
                vals_item = [
                    np.dot(beta_s_item, np.array(feats[item_id][image_list[k]]).reshape((1,10)).T)
                    for k in range(len(image_list))]
                vals_item = np.array(vals_item).reshape(len(image_list))
                #creative-wise value
                vals_c = [
                    np.dot(beta_s_c[k], np.array(feats[item_id][image_list[k]]).reshape((1,10)).T)
                    for k in range(len(image_list))]
                vals_c = np.array(vals_c).reshape(len(image_list))
                #weighted sum vals_item and vals_c
                alpha = sigmoid(np.sum(image_num_pv[:,:i+1]))
                vals = (1-alpha) * vals_item + alpha * vals_c

                index = np.argmax(vals)
                if image_list[index] == clk_data[j][0]:
                    num_pv[i] += 1
                    num_clk[i] += clk_data[j][1]
                    image_num_pv[index][i]+= 1
                    image_num_clk[index][i]+=clk_data[j][1]

    return num_clk, num_pv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bandit Alogrithm.")
    parser.add_argument('--algo', type=str, default='neuralthompson', help='Which bandit algorithm to run.')
    """
        random:random_, egreedy:e_greedy,
        thompson:thompson, neuralthompson:neural_thompson,
        ucb:ucb, neuralucb:neural_ucb,
        nlitemwise:neural_linear_item_wise, nlcreativewise:neural_linear_creative_wise,
        hybridmodel: hybrid_model
    """
    parser.add_argument('--e', type=float, default=0.1, help="E_Greedy's e value.")
    parser.add_argument('--lamda', type=float, default=0.1, help="UCB's parameter control E&E.")
    parser.add_argument('--num_process', type=int, default=10, help="How many process you want to use.")
    parser.add_argument('--content-feature-path', type=str, default='extracted_feat/extract_feat_and_score.txt', help='path of extracted image representations.')
    args = parser.parse_args()

    algo = args.algo
    e = args.e
    lamda = args.lamda
    num_process = args.num_process
    num_days = 15

    items, feats = read_file(args.content_feature_path)
    item_id_list = list(items.keys())

    if algo == 'random':
        run_function = random_

    elif algo == 'egreedy':
        run_function = e_greedy

    elif algo == 'thompson':
        run_function = thompson

    elif algo == 'neuralthompson':
        run_function = neural_thompson

    elif algo == 'ucb':
        run_function = ucb

    elif algo == 'neuralucb':
        run_function = neural_ucb

    elif algo == 'nlitemwise':
        run_function = neural_linear_item_wise

    elif algo == 'nlcreativewise':
        run_function = neural_linear_creative_wise

    elif algo == 'hybridmodel':
        run_function = hybrid_model

    else:
        raise(Exception("Please examine your algo parameter."))

    #item_id_list = item_id_list[:60]
    if num_process>=2:
        num_of_item_id = len(item_id_list)
        num_item_per_process = num_of_item_id//num_process

        args_list = []

        for i in range(num_process):
            if i<num_process-1:
                args_list.append(item_id_list[i*num_item_per_process:(i+1)*num_item_per_process])
            else:
                args_list.append(item_id_list[i*num_item_per_process:])

        pool = Pool(processes=num_process)

        returnList = pool.map(run_function, args_list)

        total_pv = np.zeros(num_days)
        total_clk = np.zeros(num_days)

        for i in range(num_process):
            for j in range(num_days):
                total_clk[j] += returnList[i][0][j]
                total_pv[j] += returnList[i][1][j]
        pool.close()
        pool.join()

        print ('The final SCTR is: ', np.sum(total_clk)/np.sum(total_pv))
        #print(list(map(int, total_clk.tolist())))
        #print(list(map(int, total_pv.tolist())))
    else:
        total_clk,total_pv = run_function(item_id_list)
        print ('The final SCTR is: ', np.sum(total_clk)/np.sum(total_pv))
        #print(list(map(int, total_clk.tolist())))
        #print(list(map(int, total_pv.tolist())))
