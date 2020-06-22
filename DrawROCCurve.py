import os
import random
from utils.help_utils import pickle_load_data, get_entity_to_id_dict, get_relation_to_id_dict, check_dir, \
    pickle_save_data, json_save_data, draw_roc_curve, draw_precision_recall_curve, get_top_predictions_stat, \
    get_id_to_entity_dict
from openke.config import Tester
from openke.module.model import TransE
from openke.data import TrainDataLoader, TestDataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
from sklearn import metrics
from time import time
import memory_profiler


dataset_name = 'Wikidata_7concepts_test=exact_all_add_reverse'
model = 'transe'
negative_mode = 'normal'
triple_per_report = 10000
total_triple = 73043896

checkpoint_path = './checkpoint/' + model + '/' + dataset_name + '/'

data_path = "./benchmarks/" + dataset_name + "/"
pra_relation_to_triples_dict = pickle_load_data(os.path.join(data_path, 'relation_to_triples_dict.pkl'))

# settings
train_times = 1000
vec_dim = 512
opt_method = 'sgd'
momentum = 0.9

margin = 15
learning_rate = 0.1
weight_decay = 0
p_norm = 1
use_gpu = True

result_name = 'vec_dim=' + str(vec_dim) + '_margin=' + str(margin) + '_train_times=' + str(train_times) \
              + '_opt_method=' + opt_method + '_learning_rate=' + str(learning_rate) \
              + '_momentum=' + str(momentum) + '_weight_decay=' + str(weight_decay) \
              + '_p_norm=' + str(p_norm) + '_negative_mode=' + negative_mode + '_use_gpu=' + str(use_gpu)

#TODO rename this
negative_sampling_mode = 'none'
if negative_sampling_mode == 'Sqr':
    roc_folder_name = 'ROC_Sqr_negative_samples'
elif negative_sampling_mode == 'tri':
    roc_folder_name = 'ROC_tri_negative_samples'
else:
    random_negative = False
    roc_folder_name = 'ROC_all_negative_samples_random' if random_negative else 'ROC_all_negative_samples'

roc_result_path = os.path.join(checkpoint_path, roc_folder_name, result_name)
check_dir(roc_result_path)
by_relation_result_path = os.path.join(roc_result_path, 'by relation')
check_dir(by_relation_result_path)

relation_to_triples_dict_path = os.path.join(roc_result_path, 'relation_to_triples_dict.pkl')
roc_curve_path = os.path.join(roc_result_path, 'ROC curve')
roc_curve_by_relations_path = os.path.join(by_relation_result_path, 'ROC')

precision_recall_curve_path = os.path.join(roc_result_path, 'Precision-Recall curve')
precision_recall_curve_by_relations_path = os.path.join(by_relation_result_path, 'PR')
result_path = os.path.join(roc_result_path, 'result.txt')

entity_to_id_dict = get_entity_to_id_dict(os.path.join(data_path, 'entity2id.txt'))
id_to_entity_dict = get_id_to_entity_dict(os.path.join(data_path, 'entity2id.txt'))
relation_to_id_dict = get_relation_to_id_dict(os.path.join(data_path, 'relation2id.txt'))
entity_num = len(entity_to_id_dict)

print('-' * 100)
print(result_name)
relation_to_triples_dict = {}
if not os.path.exists(relation_to_triples_dict_path):
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=data_path,
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0
    )

    # dataloader for test
    test_dataloader = TestDataLoader(data_path, "link")

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=vec_dim,
        p_norm=1,
        norm_flag=True
    )

    # test the model
    transe.load_checkpoint(os.path.join(checkpoint_path, result_name))
    tester = Tester(
        model=transe,
        data_loader=test_dataloader,
        use_gpu=use_gpu
    )

    triple_count = 0
    t1 = time()

    for i, relation in enumerate(pra_relation_to_triples_dict):
        relation_to_triples_dict[relation] = []

        if negative_sampling_mode == 'tri':  # todo skip false negatives
            for pra_triple in pra_relation_to_triples_dict[relation]:
                if pra_triple['label'] == 0:  # only generate negative samples for positive ones
                    continue
                # positive sample
                test_data = {
                    'batch_h': np.array([entity_to_id_dict[pra_triple['head']]]),
                    'batch_t': np.array([entity_to_id_dict[pra_triple['tail']]]),
                    'batch_r': np.array([relation_to_id_dict[pra_triple['relation']]]),
                    'mode': 'tail_batch'
                }
                res = tester.test_one_step(test_data)

                # cur_triple = deepcopy(pra_triple)
                cur_triple = pra_triple
                cur_triple['score'] = -res[0]
                cur_triple['confidence'] = -res[0]
                relation_to_triples_dict[relation].append(cur_triple)

                # add negative samples
                # TODO skip false negatives
                for entity_index in range(entity_num):
                    selected_index = int(entity_index * entity_index * entity_index * random.random())
                    if entity_index * entity_index * entity_index > entity_num:
                        break
                    test_data = {
                        'batch_h': np.array([entity_to_id_dict[pra_triple['head']]]),
                        'batch_t': np.array([selected_index]),
                        'batch_r': np.array([relation_to_id_dict[pra_triple['relation']]]),
                        'mode': 'tail_batch'
                    }
                    res = tester.test_one_step(test_data)

                    cur_triple = {
                        'relation': relation,
                        'label': 0,
                        'head': cur_triple['head'],
                        'tail': id_to_entity_dict[selected_index],
                        'score': -res[0],
                        'confidence': -res[0]
                    }
                    relation_to_triples_dict[relation].append(cur_triple)
        else:
            for pra_triple in pra_relation_to_triples_dict[relation]:
                triple_count += 1
                if random_negative:
                    # todo skip false negatives
                    test_data = {
                        'batch_h': np.array([entity_to_id_dict[pra_triple['head']]]),
                        'batch_t': np.array([entity_to_id_dict[pra_triple['tail']] if pra_triple['label'] == 1 else random.randint(0, 348235)]),
                        'batch_r': np.array([relation_to_id_dict[pra_triple['relation']]]),
                        'mode': 'tail_batch'
                    }
                else:
                    test_data = {
                        'batch_h': np.array([entity_to_id_dict[pra_triple['head']]]),
                        'batch_t': np.array([entity_to_id_dict[pra_triple['tail']]]),
                        'batch_r': np.array([relation_to_id_dict[pra_triple['relation']]]),
                        'mode': 'tail_batch'
                    }

                res = tester.test_one_step(test_data)

                # cur_triple = deepcopy(pra_triple)
                cur_triple = pra_triple
                cur_triple['score'] = -res[0]
                cur_triple['confidence'] = -res[0]
                relation_to_triples_dict[relation].append(cur_triple)

                if triple_count % triple_per_report == 0:
                    print(triple_count, 'time left', (time() - t1) / triple_count * (total_triple - triple_count) / 3600,
                          'mem used', memory_profiler.memory_usage()[0] / 1024)

        print(i, relation, 'len(relation_to_triples_dict[relation]', len(relation_to_triples_dict[relation]))

    pickle_save_data(relation_to_triples_dict_path, relation_to_triples_dict)
else:
    relation_to_triples_dict = pickle_load_data(relation_to_triples_dict_path)

labels = []
confidences = []
actual_positive_num = 0
actual_negative_num = 0
for i, relation in enumerate(relation_to_triples_dict):
    print(i, relation)
    true_positives_num = 0
    false_positives_num = 0
    cur_relation_triples = relation_to_triples_dict[relation]
    cur_relation_triples = sorted(cur_relation_triples, key=lambda t: t['confidence'], reverse=True)

    cur_relation_labels = []
    cur_relation_confidences = []
    for triple in cur_relation_triples:
        labels.append(triple['label'])
        confidences.append(triple['confidence'])
        cur_relation_labels.append(triple['label'])
        cur_relation_confidences.append(triple['confidence'])

        if triple['label'] == 1:
            actual_positive_num += 1
            # print(triple['confidence'])
        else:
            actual_negative_num += 1

    if len(cur_relation_labels) == 0:
        continue
    draw_roc_curve(np.array(cur_relation_labels), np.array(cur_relation_confidences),
                   roc_curve_by_relations_path + '_' + relation)
    draw_precision_recall_curve(np.array(cur_relation_labels), np.array(cur_relation_confidences),
                                precision_recall_curve_by_relations_path + '_' + relation)
    get_top_predictions_stat(cur_relation_confidences, cur_relation_labels)
    # if relation == 'cast member':
    #     print('true_positives_num', true_positives_num, 'false_positives_num', false_positives_num)
    #     exit(-1)

roc_auc, optimal_threshold = draw_roc_curve(np.array(labels), np.array(confidences), roc_curve_path)
average_precision = draw_precision_recall_curve(np.array(labels), np.array(confidences), precision_recall_curve_path)

print('-' * 100)
top_predictions_stat = get_top_predictions_stat(confidences, labels)
print('roc_auc', roc_auc)
print('optimal_threshold_by_roc', optimal_threshold)
print('average_precision', average_precision)
# print('optimal_threshold_precision_score', precision_score(labels, predicted_labels))
# print('optimal_threshold_recall_score', recall_score(labels, predicted_labels))
print('test samples num', len(labels))
print('actual_positive_num', actual_positive_num)
print('actual_negative_num', actual_negative_num)
result_dict = {
    'optimal_threshold_by_roc': float(optimal_threshold),
    'average_precision': float(average_precision),
    'roc_auc': roc_auc,
    # 'optimal_threshold_precision_score': precision_score(labels, predicted_labels),
    # 'optimal_threshold_recall_score': recall_score(labels, predicted_labels),
    'test_samples_num': len(labels),
    'actual_positive_num': actual_positive_num,
    'actual_negative_num': actual_negative_num,
    'top_predictions_stat': top_predictions_stat
}
json_save_data(result_path, result_dict)