import argparse
import importlib
import random
import yaml

import numpy as np

import mlflow
import torch
import torch.utils.data
import tensorflow as tf
from tensorboardX import SummaryWriter

from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

import training
import settings
from utils import save_checkpoint, load_checkpoint
from newsgroups.cbs import generate_cbs_features
import torchvision.transforms as transforms
from reproduction import lib_data


parser = argparse.ArgumentParser(description='Main File')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

# parser.add_argument('--dataset', type=str, default='mnist',
#                     help='Name of dataset: mnist, cifar10 or newsgroups')

parser.add_argument('--random-seed', type=int, default=None)
parser.add_argument('--params-path', type=str, default=None)
parser.add_argument('--ppe-save-path', type=str, default=None)
parser.add_argument('--ppe-load-path', type=str, default=None)

parser.add_argument("--id_dataset", required=True)
parser.add_argument("--ood_dataset", required=True)
parser.add_argument("--experiment_name", required=True)
parser.add_argument(
    "-wp",
    "--with_param",
    type=str,
    action="append",
    default=[],
    help="Optional repeated argument of the form k=[v], "
         "will be included in the cartesian product of parameters, "
         "using k as the gin parameter name. "
         "Example usage: --with_param data.batch_size=[32,64,128]",
)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


# XXX: Load model architecture and data sets.
if "mnist" in args.id_dataset:
    prepare_data = importlib.import_module("mnist.pu_biased_n")
    params = prepare_data.params
    Net = prepare_data.Net
else:
    # TODO: pick one of mammalP and transportP to move in cifar10
    prepare_data = importlib.import_module("cifar10.pu_biased_n")
    params = prepare_data.params
    Net = prepare_data.Net
# if args.dataset == 'newsgroups':
#     NetCBS = prepare_data.NetCBS
# train_data_orig = prepare_data.train_data
# test_data_orig = prepare_data.test_data
# train_labels_orig = prepare_data.train_labels
# test_labels = prepare_data.test_labels


# XXX: parse parameters (from param file and from the file loaded for the dataset).
if args.params_path is not None:
    with open(args.params_path) as f:
        params_file = yaml.load(f)
    for key in params_file:
        params[key] = params_file[key]


# Parse params passed as command line arguments.
for param_string in args.with_param:
    [k, v] = param_string.split("=")
    v = eval(v)  # Risky but what you gonna do about it.
    params[k] = v
    params[f"\n{k}"] = v


if args.random_seed is not None:
    params['\nrandom_seed'] = args.random_seed

if args.ppe_save_path is not None:
    params['\nppe_save_name'] = args.ppe_save_path

if args.ppe_load_path is not None:
    params['ppe_load_name'] = args.ppe_load_path


num_classes = params['num_classes']

p_num = params['\np_num']
n_num = params.get('n_num', 0)
sn_num = params['sn_num']
u_num = params['u_num']

pv_num = params['\npv_num']
nv_num = params.get('nv_num', 0)
snv_num = params['snv_num']
uv_num = params['uv_num']

u_cut = params['\nu_cut']

pi = params['\npi']
rho = params['rho']
true_rho = params.get('true_rho', rho)

positive_classes = params['\npositive_classes']
negative_classes = params.get('negative_classes', None)
neg_ps = params['neg_ps']

non_pu_fraction = params['\nnon_pu_fraction']  # gamma
balanced = params['balanced']

u_per = params['\nu_per']  # tau
adjust_p = params['adjust_p']
adjust_sn = params['adjust_sn']

cls_training_epochs = params['\ncls_training_epochs']
convex_epochs = params['convex_epochs']

p_batch_size = params['\np_batch_size']
n_batch_size = params.get('n_batch_size', 0)
sn_batch_size = params['sn_batch_size']
u_batch_size = params['u_batch_size']

learning_rate_cls = params['\nlearning_rate_cls']
weight_decay = params['weight_decay']

if 'learning_rate_ppe' in params:
    learning_rate_ppe = params['learning_rate_ppe']
else:
    learning_rate_ppe = learning_rate_cls

milestones = params.get('milestones', [1000])
milestones_ppe = params.get('milestones_ppe', milestones)
lr_d = params.get('lr_d', 1)

non_negative = params['\nnon_negative']
nn_threshold = params['nn_threshold']  # beta
nn_rate = params['nn_rate']

cbs_feature = params.get('\ncbs_feature', False)
cbs_feature_later = params.get('cbs_feature_later', False)
cbs_alpha = params.get('cbs_alpha', 10)
cbs_beta = params.get('cbs_beta', 4)
n_select_features = params.get('n_select_features', 0)
svm = params.get('svm', False)
svm_C = params.get('svm_C', 1)

pu_prob_est = params['\npu_prob_est']
use_true_post = params['use_true_post']

partial_n = params['\npartial_n']  # PUbN
hard_label = params['hard_label']

pn_then_pu = params.get('pn_then_pu', False)
pu_then_pn = params.get('pu_then_pn', False)  # PU -> PN

iwpn = params['\niwpn']
pu = params['pu']
pnu = params['pnu']
unbiased_pn = params.get('unbiased_pn', False)

random_seed = params['\nrandom_seed']

ppe_save_name = params.get('\nppe_save_name', None)
ppe_load_name = params.get('ppe_load_name', None)

log_dir = 'logs/MNIST'
visualize = False

priors = params.get('\npriors', None)
if priors is None:
    priors = [1/num_classes for _ in range(num_classes)]


settings.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor


for key, value in params.items():
    print('{}: {}'.format(key, value))
print('\nvalidation_interval', settings.validation_interval)
print('', flush=True)

# ======================================================================================================================


# def posteriors(labels):
#     posteriors = torch.zeros(labels.size())
#     for i in range(num_classes):
#         if i in positive_classes:
#             posteriors[labels == i] = 1
#         else:
#             posteriors[labels == i] = neg_ps[i] * rho * 1/priors[i]
#     return posteriors.unsqueeze(1)
#
#
# def pick_p_data(data, labels, n):
#     p_idxs = np.zeros_like(labels)
#     for i in range(num_classes):
#         if i in positive_classes:
#             p_idxs[(labels == i).numpy().astype(bool)] = 1
#     p_idxs = np.argwhere(p_idxs == 1).reshape(-1)
#     selected_p = np.random.choice(p_idxs, n, replace=False)
#     return data[selected_p], posteriors(labels[selected_p])
#
#
# def pick_u_data(data, labels, n):
#     if negative_classes is None:
#         selected_u = np.random.choice(len(data), n, replace=False)
#     else:
#         u_idxs = np.zeros_like(labels)
#         for i in range(num_classes):
#             if i in positive_classes or i in negative_classes:
#                 u_idxs[(labels == i).numpy().astype(bool)] = 1
#         u_idxs = np.argwhere(u_idxs == 1).reshape(-1)
#         selected_u = np.random.choice(u_idxs, n, replace=False)
#     return data[selected_u], posteriors(labels[selected_u])

def get_dataset_size(dataset):
    if dataset is None:
        return 0
    size = 0
    for batch_X, batch_y in dataset:
        y_shape = batch_y.shape
        size += y_shape[0] if len(y_shape) > 0 else 1
    return size

def split_train_valid(dataset, split_ratio):
    actual_dataset_size = get_dataset_size(dataset)
    split2_size = int(actual_dataset_size * split_ratio)

    if split2_size > 0:
        split2_data = dataset.take(split2_size)
    else:
        split2_data = None

    split1_data = dataset.skip(split2_size)
    return split1_data, split2_data, split2_size

# XXX: extract data sets: P, U, validation, test
# train_data_orig = prepare_data.train_data
# test_data_orig = prepare_data.test_data
# train_labels_orig = prepare_data.train_labels
# test_labels = prepare_data.test_labels

total_batch_size = 128
p_batch_size = int(pi * total_batch_size)
u_batch_size = total_batch_size - p_batch_size
valid_ratio = 0.2

# Get tf datasets.
max_dataset_size = 600000
all_p_data_orig = lib_data.load_dataset(args.id_dataset, split="train", reindex_labels=True)
p_test_data = lib_data.load_dataset(args.id_dataset, split="test", reindex_labels=True)
n_test_data = lib_data.load_dataset(args.ood_dataset, split="test", reindex_labels=True)
all_u_data_orig = p_test_data.concatenate(n_test_data)

# TODO: remove
# max_dataset_size = 5000
all_u_data_orig = all_u_data_orig.take(max_dataset_size)
all_p_data_orig = all_p_data_orig.take(max_dataset_size)

all_p_data = all_p_data_orig.shuffle(max_dataset_size, seed=0)
all_u_data = all_u_data_orig.shuffle(max_dataset_size, seed=0)

p_set, p_validation, pv_size = split_train_valid(all_p_data, split_ratio=valid_ratio)
u_set, u_validation = all_u_data, all_u_data.shuffle(max_dataset_size, seed=0).take(pv_size)

test_set = p_test_data.map(lambda x, y: (x, 1)).concatenate(
    n_test_data.map(lambda x, y: (x, 0))
).shuffle(max_dataset_size)

# Convert to torch tensors.
p_set_np = next(p_set.batch(max_dataset_size).take(1).as_numpy_iterator())
u_set_np = next(u_set.batch(max_dataset_size).take(1).as_numpy_iterator())
p_validation_np = next(p_validation.batch(max_dataset_size).take(1).as_numpy_iterator())
u_validation_np = next(u_validation.batch(max_dataset_size).take(1).as_numpy_iterator())
test_set_np = next(test_set.batch(max_dataset_size).take(1).as_numpy_iterator())

p_set = torch.Tensor(p_set_np[0]).permute([0, 3, 1, 2])
u_set = torch.Tensor(u_set_np[0]).permute([0, 3, 1, 2])
p_validation = torch.Tensor(p_validation_np[0]).permute([0, 3, 1, 2])
u_validation = torch.Tensor(u_validation_np[0]).permute([0, 3, 1, 2])
test_set = torch.Tensor(test_set_np[0]).permute([0, 3, 1, 2])
test_labels = torch.Tensor(test_set_np[1])

mlflow_params = {
    "id_dataset": args.id_dataset,
    "ood_dataset": args.ood_dataset,
    "method": "nnPU",
    "p_set_size": p_set.shape[0],
    "u_set_size": u_set.shape[0],
    "p_validation_size": p_validation.shape[0],
    "u_validation_size": u_validation.shape[0],
    "test_size": test_set.shape[0],
    "batch_size": total_batch_size,
    "pi": pi,
    "epochs": cls_training_epochs,
    "start_lr": learning_rate_cls,
    "nnpu_threshold": nn_threshold,
    "nnpu_stepsize": nn_rate,
}

print("MLFLOW params", mlflow_params)

# Convert to torch datasets.
p_set = torch.utils.data.TensorDataset(p_set, torch.ones(mlflow_params["p_set_size"]))
u_set = torch.utils.data.TensorDataset(u_set, torch.zeros(mlflow_params["u_set_size"]))
p_validation = p_validation, torch.ones(mlflow_params["p_validation_size"])
u_validation = u_validation, torch.zeros(mlflow_params["u_validation_size"])
test_set = torch.utils.data.TensorDataset(test_set, test_labels)

# p_set, u_set = convert_dataset(p_set.map(lambda x, y: x)), convert_dataset(u_set.map(lambda x, y: x))
# test_set = convert_dataset(test_set.map(lambda x, y: x))
# p_validation, u_validation = convert_dataset(p_validation.map(lambda x, y: x)), convert_dataset(u_validation.map(lambda x, y: x))

# t_labels = torch.zeros(test_labels.size())
#
# for i in range(num_classes):
#     if i in positive_classes:
#         t_labels[test_labels == i] = 1
#     elif negative_classes is None or i in negative_classes:
#         t_labels[test_labels == i] = -1
#     else:
#         t_labels[test_labels == i] = 0

np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True


# idxs = np.random.permutation(len(train_data_orig))
#
# valid_data = train_data_orig[idxs][u_cut:]
# valid_labels = train_labels_orig[idxs][u_cut:]
# train_data = train_data_orig[idxs][:u_cut]
# train_labels = train_labels_orig[idxs][:u_cut]
#
#
# u_data, u_pos = pick_u_data(train_data, train_labels, u_num)
# p_data, p_pos = pick_p_data(train_data, train_labels, p_num)
# uv_data, uv_pos = pick_u_data(valid_data, valid_labels, uv_num)
# pv_data, pv_pos = pick_p_data(valid_data, valid_labels, pv_num)
# test_data = test_data_orig
#
# test_posteriors = posteriors(test_labels)
# test_idxs = np.argwhere(t_labels != 0).reshape(-1)
# test_set = torch.utils.data.TensorDataset(
#     test_data[test_idxs],
#     t_labels.unsqueeze(1).float()[test_idxs],
#     test_posteriors[test_idxs])
#
#
# XXX: need to set u_set, p_set, u_validation, p_validation, test_set
# XXX: can ignore u_pos, p_pos etc for the purpose of nnPU
#
# u_set = torch.utils.data.TensorDataset(u_data, u_pos)
# u_set = torch.utils.data.TensorDataset(test_data[test_idxs],)
# u_validation = uv_data,
#
# p_set = torch.utils.data.TensorDataset(p_data,)
# p_validation = pv_data,


# TODO: add lr as metric in mlflow

# This does nnPU.
if pu:
    lib_data.setup_mlflow()
    mlflow.set_experiment(args.experiment_name)

    run = lib_data.retry(lambda: mlflow.start_run(run_name=f"{args.id_dataset}_vs_{args.ood_dataset}"))
    run_id = run.info.run_id
    mlflow_params["run_id"] = run_id
    lib_data.retry(lambda: mlflow.log_params(mlflow_params))

    print('')
    model = Net().cuda() if args.cuda else Net()
    cls = training.PUClassifier(
            model, pi=pi, balanced=balanced,
            lr=learning_rate_cls, weight_decay=weight_decay,
            milestones=milestones, lr_d=lr_d,
            nn=non_negative, nn_threshold=nn_threshold, nn_rate=nn_rate)
    cls.train(p_set, u_set, test_set, p_batch_size, u_batch_size,
              p_validation, u_validation,
              cls_training_epochs, convex_epochs=convex_epochs)

    # retry(lambda: mlflow.log_metrics(best_result))
    lib_data.retry(lambda: mlflow.end_run())


# n_embedding_points = 500
# if visualize:
#
#     indx = np.random.choice(test_data.size(0), size=n_embedding_points)
#
#     embedding_data = test_data[indx]
#     embedding_labels = t_labels.numpy().copy()
#     # Negative data that are not sampled
#     embedding_labels[test_posteriors.numpy().flatten() < 1/2] = 0
#     embedding_labels = embedding_labels[indx]
#     features = cls.last_layer_activation(embedding_data)
#     writer = SummaryWriter(log_dir=log_dir)
#     # writer.add_embedding(embedding_data.view(n_embedding_points, -1),
#     #                      metadata=embedding_labels,
#     #                      tag='Input', global_step=0)
#     writer.add_embedding(features, metadata=embedding_labels,
#                          tag='PUbN Features',
#                          global_step=cls_training_epochs)
