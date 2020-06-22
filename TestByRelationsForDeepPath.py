import os
from openke.config import Tester
from openke.module.model import TransE
from openke.data import TrainDataLoader, TestDataLoader

dataset_name = 'FB15K237_deeppath'


checkpoint_path = './checkpoint/transe_' + dataset_name + '/'
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

data_path = "./benchmarks/" + dataset_name + "/"
result_file = os.path.join(checkpoint_path, 'result.txt')

selected_relation_path = os.path.join(data_path, 'selected_relations')
selected_relations = os.listdir(selected_relation_path)

# settings
train_times = 1000
vec_dim = 200
opt_method = 'sgd'
momentum = 0.9

margin = 5.25
learning_rate = 0.1
weight_decay = 0
p_norm = 1
reverse_edge_added = True

total_right_rr = 0
total_right_mrr = 0
relation_count = 0
total_triple_count = 0
rel_to_right_mrr_dict = {}

for selected_relation in selected_relations:
    relation_count += 1

    cur_relation_path = os.path.join(data_path, 'selected_relations', selected_relation) + "/"
    with open(os.path.join(cur_relation_path, 'test2id.txt'), 'r', encoding='utf-8') as f:
        cur_triple_count = int(f.readline())
        total_triple_count += cur_triple_count
        print('cur_relation_count', cur_triple_count)

    result_name = 'vec_dim=' + str(vec_dim) + '_margin=' + str(margin) + '_train_times=' + str(train_times) \
                  + '_opt_method=' + opt_method + '_learning_rate=' + str(learning_rate) \
                  + '_momentum=' + str(momentum) + '_weight_decay=' + str(weight_decay) + '_p_norm=' + str(p_norm)

    print('-' * 100)
    print(result_name)

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
        p_norm=1,
        norm_flag=True
    )

    # test the model
    transe.load_checkpoint(os.path.join(checkpoint_path, result_name))
    tester = Tester(
        model=transe,
        data_loader=test_dataloader,
        use_gpu=True
    )
    res = tester.run_link_prediction(type_constrain=False)  # mrr, mr, hit10, hit3, hit1

    # with open(result_file, 'a', encoding='utf-8') as f:
    #     f.write(result_name + '\t' + ' right_mrr=' + str(res[0]) + ' mr=' + str(res[1])
    #             + ' hit10=' + str(res[2]) + ' hit3=' + str(res[3]) + ' hit1=' + str(res[4]) + '\n')
    print(result_name,  'relation=', selected_relation + '\t' + ' right_mrr=' + str(res[0]) + ' mr=' + str(res[1])
          + ' hit10=' + str(res[2]) + ' hit3=' + str(res[3]) + ' hit1=' + str(res[4]) + '\n')
    total_right_mrr += res[0]
    total_right_rr += res[0] * cur_triple_count

    rel_to_right_mrr_dict[int(selected_relation)] = res[0]

print('-' * 100)
print('weighted mrr', total_right_rr / total_triple_count)
print('ave mrr', total_right_mrr / relation_count)
print('rel_to_right_mrr_dict', rel_to_right_mrr_dict)