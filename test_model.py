from openke.module.model import TransE
from openke.config import Trainer, Tester
from openke.data import TrainDataLoader, TestDataLoader
import os

in_path = "./benchmarks/SensorTrust/"
test_res_dir = os.path.join(in_path, 'test_res.txt')
raw_id_test_res_dir = os.path.join(in_path, 'raw_id_test_res.txt')
entity2id_dir = os.path.join(in_path, 'entity2id.txt')
relation2id_dir = os.path.join(in_path, 'relation2id.txt')

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = in_path,
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)


# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True
)

test_dataloader = TestDataLoader(in_path, "link")

# test the model
transe.load_checkpoint('./checkpoint/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
# tester.run_link_prediction_for_all_entities()
# tester.run_link_prediction()
res = tester.run_link_prediction_for_test()

id_to_entity_external_id_dict = {}
id_to_relation_type_dict = {}

with open(entity2id_dir, 'r', encoding='utf-8') as f:
	line_count = 0
	for line in f:
		line = line.rstrip('\n')
		line_count += 1
		if line_count == 1:
			continue

		split_line = line.split(' ')
		id_to_entity_external_id_dict[str(split_line[1])] = split_line[0]
# print(id_to_entity_external_id_dict)

with open(relation2id_dir, 'r', encoding='utf-8') as f:
	line_count = 0
	for line in f:
		line = line.rstrip('\n')
		line_count += 1
		if line_count == 1:
			continue

		split_line = line.split(' ')
		id_to_relation_type_dict[str(split_line[1])] = split_line[0]
print(id_to_relation_type_dict)

with open(test_res_dir, 'w', encoding='utf-8') as f:
	for item in res:
		head_entity_external_id = id_to_entity_external_id_dict[item[0]]
		tail_entity_external_id = id_to_entity_external_id_dict[item[1]]
		relation_type = id_to_relation_type_dict[item[2]].replace('_', ' ')

		f.write('|'.join((head_entity_external_id, tail_entity_external_id, relation_type, item[3])) + '\n')

with open(raw_id_test_res_dir, 'w', encoding='utf-8') as f:
	for item in res:
		relation_type = id_to_relation_type_dict[item[2]].replace('_', ' ')

		f.write('|'.join((item[0], item[1], relation_type, item[3])) + '\n')