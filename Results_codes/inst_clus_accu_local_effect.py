############################# Instance-topology accumulated local effect ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@polyu.edu.hk
#### Department: Centre for Smart Health, PolyU, Hong Kong
#### Attempt: Instance-cluster accumulated local effect
from linecache import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re


def pred_model(X = None, W = None, b = None, pred_g = None):
    '''
    :param X: matrix, (N, M, C)
    :param W: matrix, (C, G)
    :param b: array, (G, )
    :param pred_g: array, (N, )
    :return:
    '''
    Y = np.exp(np.mean(X @ W + b, axis = 1))
    for n_i in range(Y.shape[0]):
        Y[n_i, :] /= np.sum(Y[n_i, :])
    Y_scor = Y[:, pred_g]
    return Y_scor


def plot_icpdp(X_rd_dict = None, x_rd = None):
    '''
    :param X_rd_dict: dict
    :param x_rd: array
    :return:
    '''
    means = {}
    stds = {}
    for x_i in X_rd_dict:
        res_last = re.search(r'_([^_]*)$', x_i).group(1)
        res_fore = re.search(r'^(.*)_', x_i).group(1)
        print(res_fore)
        print(res_last)
        if res_last == 'mean':
            means[res_fore] = np.array(X_rd_dict[x_i])
        elif res_last == 'std':
            stds[res_fore] = np.array(X_rd_dict[x_i])

    plt.figure(figsize=(6, 4), dpi = 400)

    for label in means:
        mean = means[label]
        std = stds[label]

        plt.plot(x_rd, mean, linewidth=2, label=label)
        plt.fill_between(
            x_rd,
            mean - std,
            mean + std,
            alpha=0.2
        )

    plt.xlabel("Disturbance value")
    plt.ylabel("ITALE value")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def inst_topo_accu_local_effect(X = None, W = None, g_pred_arr = None, clus_label_arr = None,
                                show_mode = True, clus_name_list = None, rd_inter = None,
                                save_path = None, inst_topo_mat = None, lamda_rd = 0.9,
                                rd_sub_num = 5):
    '''
    :param X: matrix, (N, M, C)
    :param W: matrix, (C, G)
    :param g_pred_arr: array, (N, )
    :param clus_label_arr: array, (N, )
    :param show_mode: boolean
    :param clus_name_list: dict
    :param rd_inter: int
    :param save_path: str
    :param inst_topo_mat: matrix, (N, M)
    :param lamda_rd: float
    :param rd_sub_num: int
    :return:
    '''
    X_rd_dict = {}
    for label_i, name_i in enumerate(clus_name_list):
        opet_i_name = np.where(clus_label_arr == label_i)[0]
        nopt_i_name = list(set(np.arange(X.shape[1])) - set(opet_i_name))
        opet_mat = X[:, opet_i_name, :]
        nopt_mat = X[:, nopt_i_name, :]
        opet_topo_mat_mask = inst_topo_mat[:, opet_i_name]
        X_rd_dict[name_i] = [opet_mat, nopt_mat, opet_topo_mat_mask]

    Y_rd_dict = {}
    for label_i, name_i in enumerate(clus_name_list):
        inst_rd_mask = X_rd_dict[name_i][2].copy()
        score_list = []
        for rd_i in rd_inter:
            print(rd_i)
            inst_rd_mask[inst_rd_mask > (1 - rd_i)] = 1
            inst_rd_mask[inst_rd_mask < (1 - rd_i)] = 0
            inst_rd_mask = inst_rd_mask.reshape(inst_rd_mask.shape[0], inst_rd_mask.shape[1], 1)
            inst_rd_mask_mat = np.concatenate([inst_rd_mask] * X_rd_dict[name_i][0].shape[2], axis=-1)
            X_rd = (lamda_rd * (np.zeros_like(X_rd_dict[name_i][0]) + rd_i) +
                    (1 - lamda_rd) * (np.zeros_like(X_rd_dict[name_i][0]) + rd_i) * inst_rd_mask)
            X_nrd = X_rd_dict[name_i][1]
            X_com = np.concatenate([X_rd, X_nrd], axis=1)
            scor_com = pred_model(X=X_com, W=W, b=b, pred_g=g_pred_arr)
            Y_scor = np.mean(scor_com)
            score_list.append(Y_scor)

        score_mean_list = []
        score_std_list = []
        cache_list = []
        score_min_list = []
        score_max_list = []

        for num_i in range(len(score_list) - 1):
            count_i = num_i + 1
            if count_i % rd_sub_num == 0:
                print(len(cache_list))
                score_mean_list.append(np.mean(cache_list))
                score_min_list.append(np.min(cache_list))
                score_max_list.append(np.max(cache_list))
                cache_list = []
            elif count_i == (len(score_list) - 1):
                print(len(cache_list))
                cache_list.append(score_list[count_i] - score_list[count_i - 1])
                score_mean_list.append(np.mean(cache_list))
                score_min_list.append(np.min(cache_list))
                score_max_list.append(np.max(cache_list))
                cache_list = []
            else:
                cache_list.append(score_list[count_i] - score_list[count_i - 1])

        print(score_mean_list)
        accu_mean_list = []
        accu_std_list = []

        for num_i, _ in enumerate(score_mean_list):
            accu_mean_list.append(np.mean([np.sum(score_mean_list[:num_i]),
                                           np.sum(score_min_list[:num_i]), np.sum(score_max_list[:num_i])]))
            accu_std_list.append(np.std([np.sum(score_mean_list[:num_i]),
                                           np.sum(score_min_list[:num_i]), np.sum(score_max_list[:num_i])]))

        Y_rd_dict[name_i + r'_mean'] = accu_mean_list
        Y_rd_dict[name_i + r'_std'] = accu_std_list


    plot_icpdp(X_rd_dict = Y_rd_dict, x_rd = rd_inter[:int(rd_inter.shape[0] / rd_sub_num)])
    pf = pd.DataFrame(Y_rd_dict)
    pf.to_excel(save_path)
    print('yes!!!')


if __name__ == '__main__':
    np.random.seed(0)
    inst_topo_mat = np.random.randn(200, 900)
    rd_inter = np.arange(0, 1, 0.02)
    rd_sub_num = 5
    X = np.random.normal(loc=0.0, scale=0.4, size=(200, 900, 768))
    X[:, :500, :] += 0.9
    X[:, 500:700, :] -= 0.5
    X[:, 700:, :] -= 0.8
    W = np.random.normal(loc=0.0, scale=0.3, size=(768, 3))
    b = np.random.normal(loc=0.0, scale=0.2, size=(3, ))
    clus_name_list = ['Tumor', 'Non-Tumor', 'Other']
    clus_label_arr = np.array([0] * 500 + [1] * 200 + [2] * 200)
    save_path = r'F:\Pytorch_Projects\Interpretable_Learning\Results\ICALE_DHMC_Kidney.xlsx'
    g_pred_arr = [1] * 200

    inst_topo_accu_local_effect(X=X, W=W, g_pred_arr=g_pred_arr, clus_label_arr=clus_label_arr,
                                show_mode=True, clus_name_list=clus_name_list, rd_inter=rd_inter,
                                save_path = save_path, inst_topo_mat = inst_topo_mat,
                                rd_sub_num = rd_sub_num)





