# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm

from utils.help_utils import get_entity_to_id_dict, get_relation_to_id_dict, read_edge_file_to_set, \
    get_id_to_entity_dict, draw_precision_recall_curve, draw_roc_curve


class Tester(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkRawLeftMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkRawRightMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkLeftMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkRightMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkRawLeftMRR.restype = ctypes.c_float
        self.lib.getTestLinkRawRightMRR.restype = ctypes.c_float
        self.lib.getTestLinkLeftMRR.restype = ctypes.c_float
        self.lib.getTestLinkRightMRR.restype = ctypes.c_float
        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):        
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })


    def run_link_prediction_by_testing_data(self, data_path, test_by_pra_testing_file=False):
        total_labels = []
        total_confidences = []
        total_rr = 0
        total_test_num = 0
        entity_to_id_dict = get_entity_to_id_dict(os.path.join(data_path, 'entity2id.txt'))
        # id_to_entity_dict = get_id_to_entity_dict(os.path.join(data_path, 'entity2id.txt'))
        relation_to_id_dict = get_relation_to_id_dict(os.path.join(data_path, 'relation2id.txt'))
        graph_set = read_edge_file_to_set(os.path.join(data_path, 'edges'))

        testing_file_dir = os.path.join(data_path, 'testing_data')
        testing_files = os.listdir(testing_file_dir)
        for test_file_i, test_file in enumerate(testing_files):
            cur_labels = []
            cur_scores = []
            batch_t = []
            if not test_file.endswith('.data'):
                continue
            print(test_file_i)
            cur_relation = test_file.replace('.data', '')
            cur_relation_total_rr = 0
            cur_relation_total_test_num = 0
            with open(os.path.join(testing_file_dir, test_file), 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n')
                    split_line = line.split(',')
                    cur_head = split_line[2]
                    cur_tail = split_line[3]

                    # skip false negatives
                    if '-' in line and '\t'.join([cur_relation, cur_head, cur_tail]) in graph_set:
                        continue

                    if '+' in line and len(batch_t) > 0:
                        cur_scores = list(-1 * self.test_one_step({
                            'batch_h': np.array([entity_to_id_dict[cur_head]]),
                            'batch_t': np.array(batch_t) if test_by_pra_testing_file else np.array(range(len(entity_to_id_dict))),
                            'batch_r': np.array([relation_to_id_dict[cur_relation]]),
                            'mode': 'tail_batch'
                        }))

                        total_test_num += 1
                        cur_relation_total_test_num += 1

                        if not test_by_pra_testing_file:
                            cur_labels = [0] * len(entity_to_id_dict)
                            cur_labels[entity_to_id_dict[cur_tail]] = 1

                        count = zip(cur_scores, cur_labels)
                        count = sorted(count, key=lambda x: x[0], reverse=True)

                        rank = 0
                        for i, item in enumerate(count):
                            if item[1] == 1:
                                rank = i + 1
                                break

                        rr = 1 / rank if rank != 0 else 0
                        # print('rr', rr)
                        total_rr += rr
                        cur_relation_total_rr += rr
                        total_labels += cur_labels
                        cur_labels = []
                        cur_scores = []
                        batch_t = []

                    cur_labels.append(1 if '+' in line else 0)
                    batch_t.append(entity_to_id_dict[cur_tail])

            # -------------- wrap the last one --------------
            if len(cur_labels) > 0:
                count = zip(cur_scores, cur_labels)
                count = sorted(count, key=lambda x: x[0], reverse=True)

                rank = 0
                for i, item in enumerate(count):
                    if item[1] == 1:
                        rank = i + 1
                        break

                rr = 1 / rank if rank != 0 else 0
                # print('rr', rr)
                total_rr += rr
                cur_relation_total_rr += rr
                total_labels += cur_labels
            # -------------- wrap the last one --------------

            cur_mrr = cur_relation_total_rr / cur_relation_total_test_num if cur_relation_total_test_num > 0 else 0
            print(cur_relation, 'triple num', cur_relation_total_test_num, 'cur relation mrr', cur_mrr)
            print('overall mrr', (total_rr / total_test_num) if total_test_num > 0 else 0)

        print('overall mrr', total_rr / total_test_num,
              'total_samples_num', len(total_labels), 'positive_num', total_test_num)
        # average_precision = draw_precision_recall_curve(np.array(total_labels), np.array(total_confidences),
        #                                                 'prc')
        # roc_auc, optimal_threshold = draw_roc_curve(np.array(total_labels), np.array(total_confidences), 'roc')
        # print('average_precision', average_precision, 'roc_auc', roc_auc)

    def run_link_prediction(self, type_constrain = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
        for index, [data_head, data_tail] in enumerate(training_range):
            score = self.test_one_step(data_head)
            self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            #--------------------
            # print('mode', data_head['mode'],
            #       "len(data_head['batch_h']),", len(data_head['batch_h']),
            #       "len(data_head['batch_t'])", len(data_head['batch_t']),
            #       "len(data_head['batch_r'])", len(data_head['batch_r']))
            #--------------------
            score = self.test_one_step(data_tail)
            self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
            # --------------------
            # print('mode', data_tail['mode'],
            #       "len(data_tail['batch_h']),", len(data_tail['batch_h']),
            #       "len(data_tail['batch_t'])", len(data_tail['batch_t']),
            #       "len(data_tail['batch_r'])", len(data_tail['batch_r']))
            # --------------------
        self.lib.test_link_prediction(type_constrain)

        raw_left_mrr = self.lib.getTestLinkRawLeftMRR(type_constrain)
        raw_right_mrr = self.lib.getTestLinkRawRightMRR(type_constrain)
        left_mrr = self.lib.getTestLinkLeftMRR(type_constrain)
        right_mrr = self.lib.getTestLinkRightMRR(type_constrain)
        mrr = self.lib.getTestLinkMRR(type_constrain)
        mr = self.lib.getTestLinkMR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        print(hit10)

        return {
            'raw_left_mrr': raw_left_mrr,
            'raw_right_mrr': raw_right_mrr,
            'left_mrr': left_mrr,
            'right_mrr': right_mrr,
            'mrr': mrr,
            'mr': mr,
            'hit10': hit10,
            'hit3': hit3,
            'hit1': hit1
        }

    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        res_mx = 0.0
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    def run_triple_classification(self, threshlod = None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('classification')
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        for index, [pos_ins, neg_ins] in enumerate(training_range):
            res_pos = self.test_one_step(pos_ins)
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        score = np.concatenate(score, axis = -1)
        ans = np.array(ans)

        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod