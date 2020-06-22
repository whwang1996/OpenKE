import json
import os
import openke
import torch
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransH, TransD
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
from utils.help_utils import check_dir

print('torch.cuda.is_available()', torch.cuda.is_available())

dataset_name = 'Wikidata_7concepts_test=exact_all_add_reverse'
run_experiment = True
run_link_prediction_by_testing_data = False
test_by_pra_testing_file = False
model = 'transe'
negative_mode = 'normal'
num_threads = 10
print('dataset_name:', dataset_name, 'model:', model)

checkpoint_path = './checkpoint/' + model + '/' + dataset_name + '/'
check_dir(checkpoint_path)

data_path = "./benchmarks/" + dataset_name + "/"
result_file = os.path.join(checkpoint_path, 'result.txt')

# settings
train_times = 1000
vec_dim = 512
opt_method = 'sgd'
momentum = 0.9
weight_decay = 0
use_gpu = True

margins = [15]
learning_rates = [0.1]
p_norms = [1]


for p_norm in p_norms:
    for learning_rate in learning_rates:
        for margin in margins:
            result_name = 'vec_dim=' + str(vec_dim) + '_margin=' + str(margin) + '_train_times=' + str(train_times) \
                          + '_opt_method=' + opt_method + '_learning_rate=' + str(learning_rate) \
                          + '_momentum=' + str(momentum) + '_weight_decay=' + str(weight_decay) \
                          + '_p_norm=' + str(p_norm) + '_negative_mode=' + negative_mode + '_use_gpu=' + str(use_gpu)

            print('-' * 100)
            print(result_name)
            cur_checkpoint = os.path.join(checkpoint_path, result_name)

            # dataloader for training
            train_dataloader = TrainDataLoader(
                in_path=data_path,
                nbatches=100,
                threads=num_threads,
                sampling_mode=negative_mode,
                bern_flag=1,
                filter_flag=1,
                neg_ent=25,
                neg_rel=0
            )

            # dataloader for test
            test_dataloader = TestDataLoader(data_path, "link")

            # define the model
            if model == 'transe':
                trans_model = TransE(
                    ent_tot=train_dataloader.get_ent_tot(),
                    rel_tot=train_dataloader.get_rel_tot(),
                    dim=vec_dim,
                    p_norm=p_norm,
                    norm_flag=True
                )
            elif model == 'transh':
                trans_model = TransH(
                    ent_tot=train_dataloader.get_ent_tot(),
                    rel_tot=train_dataloader.get_rel_tot(),
                    dim=vec_dim,
                    p_norm=p_norm,
                    norm_flag=True
                )
            elif model == 'transd':
                trans_model = TransD(
                    ent_tot=train_dataloader.get_ent_tot(),
                    rel_tot=train_dataloader.get_rel_tot(),
                    dim_e=vec_dim,
                    dim_r=vec_dim,
                    p_norm=p_norm,
                    norm_flag=True
                )

            # define the loss function
            model = NegativeSampling(
                model=trans_model,
                loss=MarginLoss(margin=margin),
                batch_size=train_dataloader.get_batch_size()
            )

            if run_experiment:
                # train the model
                trainer = Trainer(
                    model=model,
                    data_loader=train_dataloader,
                    train_times=train_times,
                    alpha=learning_rate,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    opt_method=opt_method,
                    use_gpu=use_gpu
                )
                trainer.run()
                trans_model.save_checkpoint(cur_checkpoint)

            # test the model
            trans_model.load_checkpoint(cur_checkpoint)
            tester = Tester(
                model=trans_model,
                data_loader=test_dataloader,
                use_gpu=use_gpu
            )

            if run_link_prediction_by_testing_data:
                tester.run_link_prediction_by_testing_data(data_path, test_by_pra_testing_file=test_by_pra_testing_file)
            else:
                res = tester.run_link_prediction(type_constrain=False)  # mrr, mr, hit10, hit3, hit1

            with open(result_file, 'a', encoding='utf-8') as f:
                f.write(result_name + '\t' + json.dumps(res) + '\n')