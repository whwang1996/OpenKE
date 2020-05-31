import os

dataset_name = 'WN18'


result_path = './checkpoint/transe_' + dataset_name + '/'

best_settings = ''
max_right_mrr = -float("inf")
with open(os.path.join(result_path, 'result.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        split_line = line.split('\t')

        settings = split_line[0]
        res = split_line[1]

        # print(res.split(' '))
        right_mrr = float(res.split(' ')[1].split('=')[1])

        if right_mrr > max_right_mrr:
            max_right_mrr = right_mrr
            best_settings = settings

print('dataset_name', dataset_name)
print('best_settings', best_settings)
print('max_right_mrr', max_right_mrr)