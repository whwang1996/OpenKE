import os
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

dataset_name = 'FB15K237'


result_path = './checkpoint/transe_' + dataset_name + '/'
if not os.path.exists(result_path):
    os.mkdir(result_path)

data_path = "./benchmarks/" + dataset_name + "/"
result_file = os.path.join(result_path, 'result.txt')

# settings
train_times = 1000
vec_dim = 200
opt_method = 'sgd'
momentum = 0
weight_decay = 0

margins = [5, 5.5, 4.5]
learning_rates = [1, 0.1]
p_norms = [1]


for p_norm in p_norms:
    for learning_rate in learning_rates:
        for margin in margins:
            result_name = 'vec_dim=' + str(vec_dim) + '_margin=' + str(margin) + '_train_times=' + str(train_times) \
                          + '_opt_method=' + opt_method + '_learning_rate=' + str(learning_rate) \
                          + '_momentum=' + str(momentum) + '_weight_decay=' + str(weight_decay) \
                          + '_p_norm=' + str(p_norm)

            print('-' * 100)
            print(result_name)

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

            # define the loss function
            model = NegativeSampling(
                model=transe,
                loss=MarginLoss(margin=margin),
                batch_size=train_dataloader.get_batch_size()
            )

            # train the model
            trainer = Trainer(
                model=model,
                data_loader=train_dataloader,
                train_times=train_times,
                alpha=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                opt_method=opt_method,
                use_gpu=True
            )
            trainer.run()
            transe.save_checkpoint(os.path.join(result_path, result_name))

            # test the model
            transe.load_checkpoint(os.path.join(result_path, result_name))
            tester = Tester(
                model=transe,
                data_loader=test_dataloader,
                use_gpu=True
            )
            res = tester.run_link_prediction(type_constrain=False)  # mrr, mr, hit10, hit3, hit1

            with open(result_file, 'a', encoding='utf-8') as f:
                f.write(result_name + '\t' + 'left_mrr=' + str(res[0]) + ' right_mrr=' + str(res[1])
                        + ' mrr=' + str(res[2]) + ' mr=' + str(res[3])
                        + ' hit10=' + str(res[4]) + ' hit3=' + str(res[5]) + ' hit1=' + str(res[6]) + '\n')
