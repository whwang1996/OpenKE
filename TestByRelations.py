import os
from openke.config import Tester
from openke.module.model import TransE
from openke.data import TrainDataLoader, TestDataLoader
import json

dataset_name = 'Wikidata_7concepts'


result_path = './checkpoint/transe_' + dataset_name + '/'
if not os.path.exists(result_path):
    os.mkdir(result_path)

data_path = "./benchmarks/" + dataset_name + "/"
result_file = os.path.join(result_path, 'result.txt')
result_by_relation_file = os.path.join(result_path, 'result_by_relation.txt')

selected_relation_path = os.path.join(data_path, 'selected_relations')
selected_relation_ids = os.listdir(selected_relation_path)

# settings
train_times = 1000
vec_dim = 512
opt_method = 'sgd'
momentum = 0.9

margin = 14
learning_rate = 1
weight_decay = 0
p_norm = 1

result_name = 'vec_dim=' + str(vec_dim) + '_margin=' + str(margin) + '_train_times=' + str(train_times) \
              + '_opt_method=' + opt_method + '_learning_rate=' + str(learning_rate) \
              + '_momentum=' + str(momentum) \
    # + '_weight_decay=' + str(weight_decay) + '_p_norm=' + str(p_norm)

print('-' * 100)
print(result_name)

total_raw_right_rr = 0
total_raw_right_mrr = 0
total_right_rr = 0
total_right_mrr = 0
relation_count = 0
total_triple_count = 0
relation_to_statistics_dict = {}


for cur_relation in selected_relation_ids:
    relation_count += 1

    cur_relation_path = os.path.join(data_path, 'selected_relations', cur_relation) + "/"
    with open(os.path.join(cur_relation_path, 'test2id.txt'), 'r', encoding='utf-8') as f:
        cur_triple_count = int(f.readline())
        if cur_triple_count == 0:
            print(cur_relation, 'cur_triple_count=0')
            continue

        print('cur_relation_count', cur_triple_count)

    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=cur_relation_path,
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0
    )

    # dataloader for test
    test_dataloader = TestDataLoader(cur_relation_path, "link")

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=vec_dim,
        p_norm=p_norm,
        norm_flag=True
    )

    # test the model
    transe.load_checkpoint(os.path.join(result_path, result_name))
    tester = Tester(
        model=transe,
        data_loader=test_dataloader,
        use_gpu=True
    )
    res = tester.run_link_prediction(type_constrain=False)  # mrr, mr, hit10, hit3, hit1

    print(result_name,  'relation=', cur_relation + '\t' + json.dumps(res))
    total_raw_right_mrr += res['raw_right_mrr']
    total_raw_right_rr += res['raw_right_mrr'] * cur_triple_count

    total_right_mrr += res['right_mrr']
    total_right_rr += res['right_mrr'] * cur_triple_count
    total_triple_count += cur_triple_count

    relation_to_statistics_dict[cur_relation] = res

print('-' * 100)
print('weighted ave raw right mrr', total_raw_right_rr / total_triple_count)
print('ave raw right mrr', total_raw_right_mrr / relation_count)

print('weighted ave right mrr', total_right_rr / total_triple_count)
print('ave right mrr', total_right_mrr / relation_count)
print('rel_to_right_mrr_dict', relation_to_statistics_dict)

with open(result_by_relation_file, 'a', encoding='utf-8') as f:
    f.write(result_name + '\t' + json.dumps(relation_to_statistics_dict) + '\n')