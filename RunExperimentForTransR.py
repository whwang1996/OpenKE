import json
import os

import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransR
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
from utils.help_utils import check_dir

dataset_name = 'Wikidata_7concepts'


checkpoint_path = './checkpoint/transr/' + dataset_name + '/'
check_dir(checkpoint_path)

data_path = "./benchmarks/" + dataset_name + "/"
result_file = os.path.join(checkpoint_path, 'result.txt')

# settings
train_times = 1000
vec_dim = 256
opt_method = 'sgd'
momentum = 0.9
weight_decay = 0
use_gpu = False


margins = [7.5, 7, 8, 6.5, 8.5]
learning_rates = [1, 0.1, 0.001]
p_norms = [1, 2]


for p_norm in p_norms:
    for learning_rate in learning_rates:
        for margin in margins:
            result_name = 'vec_dim=' + str(vec_dim) + '_margin=' + str(margin) + '_train_times=' + str(train_times) \
                          + '_opt_method=' + opt_method + '_learning_rate=' + str(learning_rate) \
                          + '_momentum=' + str(momentum) + '_weight_decay=' + str(weight_decay) \
                          + '_p_norm=' + str(p_norm) + '_use_gpu=' + str(use_gpu)

            print('-' * 100)
            print(result_name)
            cur_checkpoint = os.path.join(checkpoint_path, result_name)

            # dataloader for training
            train_dataloader = TrainDataLoader(
                in_path = data_path,
                nbatches = 100,
                threads = 50,
                sampling_mode = "normal",
                bern_flag = 1,
                filter_flag = 1,
                neg_ent = 25,
                neg_rel = 0)

            # dataloader for test
            test_dataloader = TestDataLoader(
                in_path = data_path,
                sampling_mode = 'link')

            # define the model
            transe = TransE(
                ent_tot = train_dataloader.get_ent_tot(),
                rel_tot = train_dataloader.get_rel_tot(),
                dim = vec_dim,
                p_norm = p_norm,
                norm_flag = True)

            model_e = NegativeSampling(
                model = transe,
                loss = MarginLoss(margin = margin),
                batch_size = train_dataloader.get_batch_size())

            transr = TransR(
                ent_tot = train_dataloader.get_ent_tot(),
                rel_tot = train_dataloader.get_rel_tot(),
                dim_e = vec_dim,
                dim_r = vec_dim,
                p_norm = p_norm,
                norm_flag = True,
                rand_init = False)

            model_r = NegativeSampling(
                model = transr,
                loss = MarginLoss(margin = margin),
                batch_size = train_dataloader.get_batch_size()
            )

            # pretrain transe
            trainer = Trainer(
                model = model_e,
                data_loader = train_dataloader,
                train_times = 1,
                alpha = 0.5,
                use_gpu = use_gpu
            )
            trainer.run()
            parameters = transe.get_parameters()
            # transe.save_parameters("./result/transr_transe.json")

            # train transr
            transr.set_parameters(parameters)
            trainer = Trainer(
                model = model_r,
                data_loader = train_dataloader,
                train_times = train_times,
                alpha = learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                opt_method=opt_method,
                use_gpu = use_gpu
            )
            trainer.run()
            transr.save_checkpoint(cur_checkpoint)

            # test the model
            transr.load_checkpoint(cur_checkpoint)
            tester = Tester(model = transr, data_loader = test_dataloader, use_gpu = use_gpu)
            res = tester.run_link_prediction(type_constrain = False)

            with open(result_file, 'a', encoding='utf-8') as f:
                f.write(result_name + '\t' + json.dumps(res) + '\n')