def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from model import HCL
from data_loader import *
import argparse
import numpy as np
import torch
import random
import sklearn.metrics as skm
import torch_geometric
from tree_utli import HRN, HRNEncoder


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_type', type=str, default='ad', choices=['oodd', 'ad'])
    parser.add_argument('-DS', help='Dataset', default='BZR')
    parser.add_argument('-DS_ood', help='Dataset', default='COX2')
    parser.add_argument('-DS_pair', default=None)
    parser.add_argument('-rw_dim', type=int, default=16)
    parser.add_argument('-dg_dim', type=int, default=16)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-batch_size_test', type=int, default=9999)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-num_layer', type=int, default=5)
    parser.add_argument('-hidden_dim', type=int, default=32)
    parser.add_argument('-num_trial', type=int, default=5)
    parser.add_argument('-num_epoch', type=int, default=400)
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-is_adaptive', type=int, default=1)
    parser.add_argument('-num_cluster', type=int, default=2)
    parser.add_argument('-alpha', type=float, default=0)

    ## tree parameters
    parser.add_argument('-l', '--local', dest='local', action='store_const',
            const=True, default=False)
    parser.add_argument('-g', '--glob', dest='glob', action='store_const',
            const=True, default=False)
    parser.add_argument('-p', '--prior', dest='prior', action='store_const',
            const=True, default=False)
    parser.add_argument('--loss_sym', action='store_true')
    parser.add_argument('--tree_depth', type=int, default=5)
    parser.add_argument('--tree_pooling_type', type=str, default='sum')
    parser.add_argument('--tree_hidden_dim', type=int, default=32)
    parser.add_argument('--tree_dropout', type=int, default=0)
    parser.add_argument('--tree_link_input', action='store_true')
    parser.add_argument('--tree_drop_root', action='store_true')
    parser.add_argument('--tree_learning_rate', type=float, default=0.01)

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)


if __name__ == '__main__':
    setup_seed(0)
    args = arg_parse()

    if args.exp_type == 'ad':
        if args.DS.startswith('Tox21'):
            dataloader, dataloader_test, meta = get_ad_dataset_Tox21(args)
        else:
            splits = get_ad_split_TU(args, fold=args.num_trial)

    aucs = []
    for trial in range(args.num_trial):
        setup_seed(trial + 1)

        if args.exp_type == 'oodd':
            dataloader, _, meta = get_ood_dataset(args, pre=True)
            _, dataloader_test, _ = get_ood_dataset(args)

        elif args.exp_type == 'ad' and not args.DS.startswith('Tox21'):
            dataloader, _, meta = get_ad_dataset_TU(args, splits[trial], pre=True)
            _, dataloader_test, _ = get_ad_dataset_TU(args, splits[trial])

        dataset_num_features = meta['num_feat']
        tree_input_dim = meta['deg_x']
        print(f"tree_input_dim: {tree_input_dim}")
        n_train = meta['num_train']

        if trial == 0:
            print('================')
            print('Exp_type: {}'.format(args.exp_type))
            print('DS: {}'.format(args.DS_pair if args.DS_pair is not None else args.DS))
            print('num_features: {}'.format(dataset_num_features))
            print('num_structural_encodings: {}'.format(args.dg_dim + args.rw_dim))
            print('hidden_dim: {}'.format(args.hidden_dim))
            print('num_gc_layers: {}'.format(args.num_layer))
            print('================')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HCL(args.hidden_dim, args.num_layer, dataset_num_features, args.dg_dim+args.rw_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        encoder = HRNEncoder(args.tree_depth, args.tree_pooling_type,
                         tree_input_dim, args.tree_hidden_dim,
                         args.hidden_dim*args.num_layer, args.tree_dropout,
                         args.tree_link_input, args.tree_drop_root,
                         device)
        treeM = HRN(encoder, args.hidden_dim*args.num_layer).to(device)
        treeOpt = torch.optim.Adam(treeM.parameters(), lr=args.tree_learning_rate)

        for epoch in range(1, args.num_epoch + 1):
            if args.is_adaptive:
                if epoch == 1:
                    weight_b, weight_g, weight_n = 1, 1, 1
                else:
                    weight_b, weight_g, weight_n = std_b ** args.alpha, std_g ** args.alpha, std_n ** args.alpha
                    weight_sum = (weight_b  + weight_g  + weight_n) / 3
                    weight_b, weight_g, weight_n = weight_b/weight_sum, weight_g/weight_sum, weight_n/weight_sum

            model.train()
            treeM.train()

            loss_all = 0
            if args.is_adaptive:
                loss_t_all, loss_g_all, loss_n_all = [], [], []

            for data in dataloader:
                data = data.to(device)
                optimizer.zero_grad()
                treeOpt.zero_grad()

                b, g_f, g_s, n_f, n_s = model(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
                x_hrn = treeM(data)
                loss_g = model.calc_loss_g(g_f, g_s)
                loss_n = model.calc_loss_n(n_f, n_s, data.batch)
                loss_t = model.calc_loss_tree(x_hrn, g_f)

                if args.is_adaptive:
                    loss = weight_b * loss_t.mean() + weight_g * loss_g.mean() + weight_n * loss_n.mean()
                    loss_t_all = loss_t_all + loss_t.detach().cpu().tolist()
                    loss_g_all = loss_g_all + loss_g.detach().cpu().tolist()
                    loss_n_all = loss_n_all + loss_n.detach().cpu().tolist()
                else:
                    loss = loss_t.mean() + loss_g.mean() + loss_n.mean()

                loss_all += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
            print('[TRAIN] Epoch:{:03d} | Loss:{:.4f}'.format(epoch, loss_all / n_train))

            if args.is_adaptive:
                mean_b, std_b = np.mean(loss_t_all), np.std(loss_t_all)
                mean_g, std_g = np.mean(loss_g_all), np.std(loss_g_all)
                mean_n, std_n = np.mean(loss_n_all), np.std(loss_n_all)

            if epoch % args.eval_freq == 0:

                model.eval()
                treeM.eval()

                y_score_all = []
                y_true_all = []
                for data in dataloader_test:
                    data = data.to(device)
                    b, g_f, g_s, n_f, n_s = model(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
                    y_score_g = model.calc_loss_g(g_f, g_s)
                    y_score_n = model.calc_loss_n(n_f, n_s, data.batch)
                    y_score_t = model.calc_loss_g(g_f, g_s)

                    if args.is_adaptive:
                        y_score = (y_score_t - mean_b)/std_b + (y_score_g - mean_g)/std_g + (y_score_n - mean_n)/std_n
                    else:
                        y_score = y_score_t + y_score_g + y_score_n
                    y_true = data.y

                    y_score_all = y_score_all + y_score.detach().cpu().tolist()
                    y_true_all = y_true_all + y_true.detach().cpu().tolist()

                auc = skm.roc_auc_score(y_true_all, y_score_all)

                print('[EVALIDATION] Epoch: {:03d} | AUC:{:.4f}'.format(epoch, auc))

        print('[RESULT] Trial: {:02d} | AUC:{:.4f}'.format(trial, auc))
        aucs.append(auc)

    avg_auc = np.mean(aucs) * 100
    std_auc = np.std(aucs) * 100
    print('[FINAL RESULT] AVG_AUC:{:.2f}+-{:.2f}'.format(avg_auc, std_auc))
