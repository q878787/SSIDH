import argparse
from datetime import datetime
import models.alexnet as alexnet
from loguru import logger

from basic_model import train_basic_data
from incre_model import train_incra_data
from models.ssidh_loss import SSIDH_Loss
from data.dataset_loader import *
from models.center_loss import CenterLoss
from utils.utils import *
import warnings
warnings.filterwarnings("ignore")

#参数预置
def load_config():
    parser = argparse.ArgumentParser(description='SSIDH_PyTorch')
    parser.add_argument('--dataset', default="cifar-10", help='Dataset name.')
    parser.add_argument('--root', default="./datasets/cifar-10/", help='Path of dataset')
    # parser.add_argument('--dataset', default="imagenet-100", help='Dataset name.')
    # parser.add_argument('--root', default="./datasets/imagenet-100/", help='Path of dataset')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate.(default: 1e-4)')
    parser.add_argument('--num-workers', default=0, type=int, help='Number of loading data threads.(default: 0)')
    parser.add_argument('--topk', default=5000, type=int, help='Calculate map of top k.(default: all)')
    # Usually adjustment
    parser.add_argument('--alpha', default=300, type=float, help='Hyper-parameter.(default: 1.0)')
    parser.add_argument('--max-iter-seen', default=150, type=int, help='Number of iterations.(default: 60)')
    parser.add_argument('--max-iter-unseen', default=400, type=int, help='Number of iterations.(default: 60)')
    parser.add_argument('--sample-percent', default=0.1, type=float, help='在已训练过的数据中重新采样的比例')
    parser.add_argument('--num-samples', default=2500, type=int, help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--batch-size', default=250, type=int, help='Batch size.(default: 64)')
    parser.add_argument('--code-length', default=64, type=int, help='Binary hash code length.(default: 12)')
    parser.add_argument('--num-seen', default=7, type=int, help='未增量前的类别数')
    parser.add_argument('--mark-percent', default=0.5, type=float, help='标注数据/总数据的比例')
    parser.add_argument('--is-incre', default=True, type=bool)
    parser.add_argument('--version', default="v1", type=str)
    parser.add_argument('--gpu', default=1, type=int, help='Using gpu.(default: -1)')

    args = parser.parse_args()
    #初始化
    if args.dataset == 'cifar-10':
        args.data_num = 59000
        args.classes_num = 10
        args.dataset_name = 'cf'
    elif args.dataset == 'imagenet-100':
        args.data_num = 100000
        args.classes_num = 100
        args.dataset_name = 'im'

    args.version = 'c' + str(args.num_seen) + '-p' + str(int(args.mark_percent*10)) + \
                   '-b'+str(args.code_length) + '-n' + str(args.num_samples) + '-bz' + \
                   str(args.batch_size)+'-' + args.dataset_name + '-' + args.version

    # GPU
    if args.gpu == -1:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)
    return args

#运行程序的入口
def run():
    args = load_config()
    logger.add('logs/'+args.version+'.log', rotation='500 MB', level='INFO')
    logger.info(args)
    logger.info(args.version)
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(6)
    #1.数据预处理--------------------------------------------------------------
    os.makedirs(os.path.join('iteratorData', args.version,"basic","final"), exist_ok=True)
    #未增量数据的基路径
    basic_path = os.path.join('iteratorData', args.version,"basic","final")
    # 训练模型之前，需要预加载omega以及类中心，因为未增量和增量数据必须统一，且类中心也要一致
    #数据分割与加载
    try:
        omegas = np.load(os.path.join(basic_path, 'omega.npy'), allow_pickle=True)
        print("加载omega···")
    except Exception:
        omegas = get_omegas(args.data_num, args.classes_num, args.num_seen, args.mark_percent)
        np.save(os.path.join(basic_path, 'omega.npy'), omegas)
        print("保存omega···")
    query_dataloader, seen_mark_dataloader, seen_unmark_dataloader, unseen_mark_dataloader, unseen_unmark_dataloader, retrieval_dataloader \
        = load_data(args.dataset, args.root, args.batch_size, args.num_workers, omegas)

    #类中心生成与加载
    center_loss = CenterLoss(args.classes_num, args.code_length)
    try:
        center_hash = torch.load(os.path.join(basic_path, 'center_hash.t'))
        center_loss.set_center_hash(center_hash)
        print("加载哈达玛类中心···")
    except Exception:
        torch.save(center_loss.get_center_hash(), os.path.join(basic_path, 'center_hash.t'))
        print("保存哈达玛类中心···")

    #CNN与Loss Function 加载
    model = alexnet.load_model(args.code_length).to(args.device)
    criterion = SSIDH_Loss(center_loss, args.code_length, args.alpha)

    #2.半监督未增量数据学习--------------------------------------------------------------
    SEEN_MARK_B = None
    if not args.is_incre:
        print("basic data train.....")
        SEEN_MARK_B, SEEN_UNMARK_B, all_seen_unmark_facker_targets, all_hit_seen_unmark_omega_index = \
            train_basic_data(model, criterion, query_dataloader, seen_mark_dataloader, seen_unmark_dataloader,
                             retrieval_dataloader, args)

    #3.半监督增量数据学习--------------------------------------------------------------
    os.makedirs(os.path.join('iteratorData', args.version, "incre", "final"), exist_ok=True)
    # 加载未增量模型生成的 标注和未标注数据的 哈希数据库，以及模型参数
    if SEEN_MARK_B is None:
        print("加载basic数据.....")
        SEEN_MARK_B = torch.load(os.path.join(basic_path, 'seen_mark_b.t'))
        SEEN_UNMARK_B = torch.load(os.path.join(basic_path, 'seen_unmark_b.t'))
        all_seen_unmark_facker_targets = torch.load(os.path.join(basic_path, 'all_seen_unmark_facker_targets.t')).float()
        all_hit_seen_unmark_omega_index = torch.load(os.path.join(basic_path, 'all_hit_seen_unmark_omega_index.t'))
        #model_dist = torch.load(os.path.join(basic_path, 'model.pt'),map_location={'cuda:3': 'cuda:0', 'cuda:2': 'cuda:0', 'cuda:1': 'cuda:0'})
        model_dist = torch.load(os.path.join(basic_path, 'model.pt'),map_location=torch.device('cpu'))
        # 加载未增量之前的模型参数
        model.load_state_dict(model_dist)

    # 训练增量数据
    print("incre data train.....")
    train_incra_data(model.to(args.device),
                     criterion,
                     query_dataloader, retrieval_dataloader, seen_mark_dataloader, seen_unmark_dataloader,
                     unseen_mark_dataloader, unseen_unmark_dataloader,
                     SEEN_MARK_B, SEEN_UNMARK_B, all_seen_unmark_facker_targets, all_hit_seen_unmark_omega_index,
                     args)



if __name__ == '__main__':
    run()