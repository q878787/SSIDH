import time
import torch.optim as optim
import utils.evaluate as evaluate
from loguru import logger
from data.dataset_loader import *
from utils.utils import *
import warnings
warnings.filterwarnings("ignore")

#训练增量模型
def train_incra_data(model,
                     criterion,
                     query_dataloader, retrieval_dataloader, seen_mark_dataloader,
                     seen_unmark_dataloader, unseen_mark_dataloader, unseen_unmark_dataloader,
                     SEEN_B,
                     args,isDIHN = False):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr / 10)

    # 构建该数据集包含 标注和未标注数据的   完整数据库 B  ，以及每次采样的数据 U
    all_seen_mark_targets = seen_mark_dataloader.dataset.get_onehot_targets()
    all_seen_mark_omega = seen_mark_dataloader.dataset.get_omega()
    all_seen_unmark_targets = seen_unmark_dataloader.dataset.get_onehot_targets()
    all_seen_unmark_omega = seen_unmark_dataloader.dataset.get_omega()
    #增量未标注数据
    all_unseen_mark_targets = unseen_mark_dataloader.dataset.get_onehot_targets().to(args.device)
    all_unseen_mark_omega = unseen_mark_dataloader.dataset.get_omega().to(args.device)
    all_unseen_unmark_targets = unseen_unmark_dataloader.dataset.get_onehot_targets().to(args.device)  # 真实标签
    all_unseen_unmark_omega = unseen_unmark_dataloader.dataset.get_omega().to(args.device)  # 完整数据集中 生成的统一下标

    all_seen_targets = torch.cat((all_seen_mark_targets, all_seen_unmark_targets), dim=0).to(args.device)
    all_unseen_targets = torch.cat((all_unseen_mark_targets,all_unseen_unmark_targets),dim=0).to(args.device)

    retrieval_dataloader.dataset.data = np.concatenate((seen_mark_dataloader.dataset.data, seen_unmark_dataloader.dataset.data,unseen_mark_dataloader.dataset.data, unseen_unmark_dataloader.dataset.data),axis=0)
    retrieval_dataloader.dataset.targets = np.concatenate((seen_mark_dataloader.dataset.targets, seen_unmark_dataloader.dataset.targets,unseen_mark_dataloader.dataset.targets, unseen_unmark_dataloader.dataset.targets),axis=0)
    retrieval_dataloader.dataset.omega = np.concatenate((seen_mark_dataloader.dataset.omega, seen_unmark_dataloader.dataset.omega,unseen_mark_dataloader.dataset.omega, unseen_unmark_dataloader.dataset.omega),axis=0)

    num_seen = len(SEEN_B)
    U =  torch.zeros(args.num_samples, args.code_length).to(args.device)
    SEEN_B = SEEN_B.to(args.device)
    UNSEEN_B = torch.randn(len(unseen_mark_dataloader.dataset)+len(unseen_unmark_dataloader.dataset), args.code_length).sign().to(args.device)
    B = torch.cat((SEEN_B,UNSEEN_B)).to(args.device)
    all_targets = torch.DoubleTensor(retrieval_dataloader.dataset.targets).to(args.device)
    all_omega =  torch.DoubleTensor(retrieval_dataloader.dataset.omega).to(args.device)

    total_time = time.time()
    best_map = 0
    for it in range(args.max_iter_unseen):
        iter_time = time.time()

        #SEEN_B = ((SEEN_B.to(args.device) + torch.randn(len(SEEN_B), args.code_length).to(args.device))).sign()
        #UNSEEN_B = ((UNSEEN_B + torch.randn(len(UNSEEN_B), args.code_length).to(args.device))).sign()

        train_dataloader, sample_omega,samples_omega_index = sample_dataloader(retrieval_dataloader, args.num_samples, args.batch_size, args.root, args.dataset)
        train_targets = train_dataloader.dataset.get_onehot_targets().to(args.device)
        S = calc_similarity_matrix(all_targets, train_targets)

        total_cnn_loss = 0
        itera = 0
        model.train()
        for batch, (data, targets, index) in enumerate(train_dataloader):
            itera = itera + 1
            data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
            u = model(data)
            U[index] = u.data

            if isDIHN:
                cnn_loss = criterion(u, B, S[index], index)
            else:
                cnn_loss = criterion(u, B, targets, S[index],torch.ones(len(u)).to(args.device))

            optimizer.zero_grad()
            cnn_loss.backward()
            optimizer.step()
            total_cnn_loss = total_cnn_loss + cnn_loss.item()

        #update_database->B----------------------------------------------------
        expand_U = torch.zeros(UNSEEN_B.shape).to(args.device)
        unseen_sample_in_unseen_index = samples_omega_index[samples_omega_index > num_seen] - num_seen
        unseen_sample_in_sample_index = (samples_omega_index > num_seen).nonzero()[0]
        expand_U[unseen_sample_in_unseen_index, :] = U[unseen_sample_in_sample_index, :]
        #UNSEEN_B = update_B(UNSEEN_B, U, S[:, len(SEEN_B):], args.code_length, expand_U, args.alpha)
        UNSEEN_B = batch_update_B(UNSEEN_B, S[:, len(SEEN_B):], U, expand_U, all_unseen_targets,train_targets, args)

        # expand_U2 = torch.zeros(SEEN_B.shape).to(args.device)
        # seen_sample_in_seen_index = samples_omega_index[samples_omega_index < num_seen]
        # seen_sample_in_sample_index = (samples_omega_index < num_seen).nonzero()[0]
        # expand_U2[seen_sample_in_seen_index, :] = U[seen_sample_in_sample_index, :]
        # SEEN_B = batch_update_B(SEEN_B, S[:, :len(SEEN_B)], U, expand_U2, all_seen_targets,train_targets, args)

        B = torch.cat((SEEN_B, UNSEEN_B), dim=0).to(args.device)

        logger.info('[iter:{}/{}][cnn_loss:{:.2f}][time:{:.2f}]'.format(
            it + 1, args.max_iter_unseen, total_cnn_loss / itera, time.time() - iter_time))


        # Evaluate----------------------------------------------------
        test_time = time.time()
        #增量后检索
        query_code, query_target = generate_code2(model, query_dataloader, args.code_length, args.classes_num,args.classes_num, args.device)
        #未增量检索
        query_code2, query_target2 = generate_code2(model, query_dataloader, args.code_length, args.classes_num,args.num_seen, args.device)

        mAP = evaluate.mean_average_precision(query_code,B,query_target,all_targets,args.device,topk=args.topk)
        mAP2 = evaluate.mean_average_precision(query_code2,SEEN_B,query_target2,all_seen_targets,args.device,topk=args.topk)

        prPath = os.path.join('iteratorData', args.version, "incre")
        # mAP = evaluate.calcMapAndPR(query_code, B, query_target, all_targets,args.topk, prPath,str(it)+"-all",True)
        # mAP2 = evaluate.calcMapAndPR(query_code, SEEN_B, query_target, all_seen_targets,args.topk, prPath, str(it)+"-all_seen",True)

        if mAP>best_map:
            best_map = mAP

        logger.info('[SSIDH-incre map:{:.4f},or_map:{:.4f},best_map:{:.4f}][time:{:.2f}]'.format(mAP, mAP2,best_map, time.time() - test_time))

        #保存散点图
        if it%10 == 0:
            num=2000
            union_path = os.path.join('iteratorData', args.version, "incre")
            #saveScatterImage(U, train_targets, num+1000, union_path , str(it) + "-ALL_U",args.classes_num)
            saveScatterImage(U.sign(), train_targets, num+1000, union_path , str(it) + "-ALL_B",args.classes_num)
            torch.save(U.cpu(), union_path + "/" + str(it) + "-all_u.t")
            torch.save(train_targets.int().cpu(), union_path + "/" + str(it) + "-all_u_targets.t")

        # 保存模型 信息
        union_path2 = os.path.join('iteratorData', args.version,"incre","final")
        torch.save(UNSEEN_B.cpu(), os.path.join(union_path2, 'unseen_b.t'))
        torch.save(all_unseen_targets.int().cpu(),os.path.join(union_path2, 'unseen_targets.t'))
        torch.save(SEEN_B.cpu(), os.path.join(union_path2, 'seen_b.t'))
        torch.save(all_seen_targets.int().cpu(), os.path.join(union_path2, 'seen_targets.t'))
        torch.save(model.state_dict(), os.path.join(union_path2, 'model.pt'))
        print("check")
    logger.info('Training SSIDH-base finish, time:{:.2f}'.format(time.time() - total_time))

