import pickle
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, roc_auc_score, \
    precision_recall_curve, auc, roc_curve, average_precision_score
from numpy import exp
import matplotlib.pyplot as plt


# Pickle loading and saving
def pickle_save_data(filename, data):
    try:
        pickle.dump(data, open(filename, "wb"))
    except Exception as e:
        print(e, end=" ")
        print("So we use the highest protocol.")
        pickle.dump(data, open(filename, "wb"), protocol=4)
    return True


def pickle_load_data(filename):
    try:
        mat = pickle.load(open(filename, "rb"))
    except Exception as e:
        mat = pickle.load(open(filename, "rb"))
    return mat


# json loading and saving
def json_save_data(filename, data):
    open(filename, "w", encoding='utf-8').write(json.dumps(data))
    return True


def json_load_data(filename):
    return json.load(open(filename, "r", encoding='utf-8'))


# directory
def check_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return True


def print_info(*text):
    print('[INFO]', ' '.join(str(item) for item in text))


def print_error(*text):
    print('[ERROR]', ' '.join(str(item) for item in text))


def get_entity_to_id_dict(entity2id_file_path):
    entity_to_id_dict = {}
    with open(entity2id_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            line = line.rstrip('\n')
            split_line = line.split('\t')
            entity_to_id_dict[split_line[0]] = int(split_line[1])

    return entity_to_id_dict


def get_id_to_entity_dict(entity2id_file_path):
    id_to_entity_dict = {}
    with open(entity2id_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            line = line.rstrip('\n')
            split_line = line.split('\t')
            id_to_entity_dict[int(split_line[1])] = split_line[0]

    return id_to_entity_dict


def get_relation_to_id_dict(relation2id_file_path):
    relation_to_id_dict = {}
    with open(relation2id_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            line = line.rstrip('\n')
            split_line = line.split('\t')
            relation_to_id_dict[split_line[0]] = int(split_line[1])

    return relation_to_id_dict


def draw_roc_curve(y_test, y_score, roc_curve_path):
    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(roc_curve_path)

    optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = 0.9
    optimal_threshold = thresholds[optimal_idx]
    # optimal_threshold = cutoff_youdens_j(fpr, tpr, thresholds)

    return roc_auc, optimal_threshold


def draw_precision_recall_curve(y_test, y_score, precision_recall_curve_path):
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    # if len(thresholds) > 1:
    #     print('thresholds[-1]', thresholds[-2], 'precision[-1]', precision[-2])

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(recall, precision, color='darkorange', lw=lw,
             label='Average precision = %0.2f' % average_precision)  # 假正率为横坐标，真正率为纵坐标做曲线
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(precision_recall_curve_path)

    return average_precision


def get_top_predictions_stat(confidences, labels):
    count = zip(confidences, labels)
    count = sorted(count, key=lambda x: x[0], reverse=True)

    top_predictions_stat = {}
    for rate in [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]:
        top_predictions = count[:int((1 - rate) * len(count))]
        # print(len(top_predictions))
        total_num = len(top_predictions)
        positive_num = 0
        for item in top_predictions:
            # print(item[0], item[1])
            if item[1] == 1:
                positive_num += 1
            # predicted_labels.append(item[1])

        print('rate', rate, 'positive_num / total_num', positive_num / total_num if total_num > 0 else 0)
        top_predictions_stat[rate] = positive_num / total_num if total_num > 0 else 0

    return top_predictions_stat


def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def read_edge_file_to_set(_edge_file_path):
    graph_set = set()
    with open(_edge_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            graph_set.add(line)

    return graph_set