import time
import torch.optim as optim
import utils.evaluate as evaluate
from loguru import logger
from data.dataset_loader import *
from utils.utils import *
import warnings

warnings.filterwarnings("ignore")

# 训练基准模型
def train_basic_data(model,
                     criterion,
                     query_dataloader,
                     seen_mark_dataloader,
                     seen_unmark_dataloader,
                     retrieval_dataloader,
                     args,isDIHN = False):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr / 10, )

    # 构建该数据集包含 标注和未标注数据的   完整数据库 B  ，以及每次采样的数据 U
    SEEN_MARK_B = torch.randn(len(seen_mark_dataloader.dataset), args.code_length).sign()
    SEEN_MARK_U = torch.zeros(args.num_samples//2, args.code_length)
    all_seen_mark_targets = seen_mark_dataloader.dataset.get_onehot_targets()
    all_seen_mark_omega = seen_mark_dataloader.dataset.get_omega()

    SEEN_UNMARK_B = torch.randn(len(seen_unmark_dataloader.dataset), args.code_length).sign()
    SEEN_UNMARK_U = torch.zeros(args.num_samples//2, args.code_length)
    all_seen_unmark_targets = seen_unmark_dataloader.dataset.get_onehot_targets()
    all_seen_unmark_omega = seen_unmark_dataloader.dataset.get_omega()

    SEEN_B = torch.cat((SEEN_MARK_B,SEEN_UNMARK_B),dim=0).to(args.device)
    SEEN_U = torch.cat((SEEN_MARK_U,SEEN_UNMARK_U),dim=0).to(args.device)
    all_seen_targets = torch.cat((all_seen_mark_targets,all_seen_unmark_targets),dim=0).to(args.device)
    all_seen_omega = torch.cat((all_seen_mark_omega,all_seen_unmark_omega),dim=0).to(args.device)

    retrieval_dataloader.dataset.data = np.append(seen_mark_dataloader.dataset.data, seen_unmark_dataloader.dataset.data,axis=0)
    retrieval_dataloader.dataset.targets = np.append(seen_mark_dataloader.dataset.targets, seen_unmark_dataloader.dataset.targets,axis=0)
    retrieval_dataloader.dataset.omega = np.append(seen_mark_dataloader.dataset.omega, seen_unmark_dataloader.dataset.omega,axis=0)

    total_time = time.time()
    best_map = 0
    for it in range(args.max_iter_seen):
        iter_time = time.time()

        train_seen_dataloader, sample_seen_omega,samples_omega_index = sample_dataloader(
            retrieval_dataloader, args.num_samples, args.batch_size, args.root, args.dataset)
        train_seen_targets = train_seen_dataloader.dataset.get_onehot_targets().to(args.device)
        SEEN_S = calc_similarity_matrix(all_seen_targets, train_seen_targets)

        total_cnn_loss = 0
        itera = 0
        model.train()
        for batch, (data, targets, index) in enumerate(train_seen_dataloader):
            itera = itera + 1
            data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
            u = model(data)
            SEEN_U[index] = u.data

            if isDIHN:
                cnn_loss = criterion(u, SEEN_B, SEEN_S[index], index)
            else:
                cnn_loss = criterion(u, SEEN_B, targets, SEEN_S[index],torch.ones(len(u)).to(args.device))

            optimizer.zero_grad()
            cnn_loss.backward()
            optimizer.step()
            total_cnn_loss = total_cnn_loss + cnn_loss.item()

        #update_database->B----------------------------------------------------
        SEEN_expand_U = torch.zeros(SEEN_B.shape).to(args.device)
        SEEN_expand_U[samples_omega_index, :] = SEEN_U
        # SEEN_B = update_B(SEEN_B, SEEN_U, SEEN_S, args.code_length, SEEN_expand_U, args.alpha)
        SEEN_B = batch_update_B(SEEN_B, SEEN_S, SEEN_U, SEEN_expand_U, all_seen_targets,train_seen_targets, args)

        logger.info('[iter:{}/{}][cnn_loss:{:.2f}][time:{:.2f}]'.format(
            it + 1, args.max_iter_seen, total_cnn_loss / itera, time.time() - iter_time))


        # Evaluate----------------------------------------------------
        test_time = time.time()
        query_code, query_target = generate_code2(model, query_dataloader, args.code_length, args.classes_num,args.num_seen, args.device)

        mAP = evaluate.mean_average_precision(query_code,SEEN_B,query_target,all_seen_targets,args.device,topk=args.topk)

        if mAP>best_map:
            best_map = mAP

        prPath = os.path.join('iteratorData', args.version, "basic")
        #mAP = evaluate.calcMapAndPR(query_code, SEEN_B, query_target, all_seen_targets, args.topk, prPath,str(it) + "-seen", True)
        logger.info('[SSIDH-base map:{:.4f},best_map:{:.4f}][time:{:.2f}]'.format(mAP,best_map, time.time() - test_time))

        # 保存散点图
        if it%10 == 0:
            num = 2000
            union_path = os.path.join('iteratorData', args.version, "basic")
            #saveScatterImage(SEEN_U.cpu(), train_seen_targets.cpu(), num, union_path, str(it) + "-seen_all-U", args.classes_num * args.num_seen / 10)
            saveScatterImage(SEEN_U.cpu().sign(), train_seen_targets.cpu(), num, union_path, str(it) + "-seen_all-B", args.classes_num * args.num_seen / 10)
            torch.save(SEEN_U.cpu(), union_path + "/" + str(it) + "-seen_all_u.t")
            torch.save(train_seen_targets.int().cpu(), union_path + "/" + str(it) + "-seen_all_targets.t")

        # 保存模型
        union_path2 = os.path.join('iteratorData', args.version, "basic", "final")
        torch.save(SEEN_B.cpu(), os.path.join(union_path2, 'seen_b.t'))
        torch.save(all_seen_targets.int().cpu(), os.path.join(union_path2, 'all_seen_targets.t'))
        torch.save(model.state_dict(), os.path.join(union_path2, 'model.pt'))
        print("check")
    logger.info('Training SSIDH-base finish, time:{:.2f}'.format(time.time() - total_time))
    return SEEN_B




