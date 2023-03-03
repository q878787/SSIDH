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
                     SEEN_MARK_B,SEEN_UNMARK_B, all_seen_unmark_facker_targets, all_hit_seen_unmark_omega_index,
                     args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr / 10)
    args.num_samples = args.num_samples // 2
    args.batch_size = args.batch_size // 2
    unmark_hp = 2

    #原始 有标注数据加载
    SEEN_MARK_B = SEEN_MARK_B.to(args.device)
    SEEN_MARK_U = torch.zeros(args.num_samples, args.code_length).to(args.device)
    all_seen_mark_targets = seen_mark_dataloader.dataset.get_onehot_targets().to(args.device)
    all_seen_mark_omega = seen_mark_dataloader.dataset.get_omega().to(args.device)

    #原始未标注数据加载
    SEEN_UNMARK_B = SEEN_UNMARK_B.to(args.device)
    SEEN_UNMARK_U = torch.zeros(args.num_samples, args.code_length).to(args.device)
    all_seen_unmark_real_targets = seen_unmark_dataloader.dataset.get_onehot_targets().to(args.device)  # 真实标签
    all_seen_unmark_facker_targets = all_seen_unmark_facker_targets.to(args.device)
    all_seen_unmark_omega = seen_unmark_dataloader.dataset.get_omega().to(args.device)

    #增量有标注数据
    UNSEEN_MARK_B = torch.randn(len(unseen_mark_dataloader.dataset), args.code_length).sign().to(args.device)
    UNSEEN_MARK_U = torch.zeros(args.num_samples*unmark_hp, args.code_length).to(args.device)
    all_unseen_mark_targets = unseen_mark_dataloader.dataset.get_onehot_targets().to(args.device)
    all_unseen_mark_omega = unseen_mark_dataloader.dataset.get_omega().to(args.device)

    #增量未标注数据
    UNSEEN_UNMARK_B = torch.randn(len(unseen_unmark_dataloader.dataset), args.code_length).sign().to(args.device)
    UNSEEN_UNMARK_U = torch.zeros(args.num_samples*unmark_hp, args.code_length).to(args.device)  # 采样样本的临时数据库
    all_unseen_unmark_real_targets = unseen_unmark_dataloader.dataset.get_onehot_targets().to(args.device)  # 真实标签
    all_unseen_unmark_facker_targets = (torch.zeros(len(unseen_unmark_dataloader.dataset), args.classes_num)).to(args.device)  # 伪标签
    all_unseen_unmark_omega = unseen_unmark_dataloader.dataset.get_omega().to(args.device)  # 完整数据集中 生成的统一下标
    # 这个数组记录命中的标签 在数据库中的下标 ，其中用了set去重
    all_hit_unseen_unmark_omega_index = []

    #有标注数据的 原始数据库+增量数据库 的标签
    ALL_MARK_TARGETS = torch.cat((all_seen_mark_targets,all_unseen_mark_targets),dim=0)
    #无标注数据 原始数据库+增量数据库 的 真实标签
    ALL_UNMARK_TARGETS = torch.cat((all_seen_unmark_real_targets, all_unseen_unmark_real_targets), dim=0)

    total_time = time.time()
    for it in range(args.max_iter_unseen):
        iter_time = time.time()

        UNSEEN_MARK_B = ((UNSEEN_MARK_B + torch.randn(len(unseen_mark_dataloader.dataset), args.code_length).to(args.device))).sign()
        UNSEEN_UNMARK_B[all_hit_unseen_unmark_omega_index] = ((UNSEEN_UNMARK_B[all_hit_unseen_unmark_omega_index] + torch.randn(len(all_hit_unseen_unmark_omega_index), args.code_length).to(args.device))).sign()

        # 从有标注数据的训练集中采样
        train_seen_mark_dataloader, sample_seen_mark_omega,samples_seen_mark_omega_index = sample_dataloader(
            seen_mark_dataloader, args.num_samples, args.batch_size, args.root, args.dataset)
        train_unseen_mark_dataloader, sample_unseen_mark_omega,samples_unseen_mark_omega_index = sample_dataloader(
            unseen_mark_dataloader, args.num_samples*unmark_hp, args.batch_size*unmark_hp, args.root, args.dataset)

        # 原始+增量的 有标注数据 生成相似矩阵(采样与全部)
        train_seen_mark_targets = train_seen_mark_dataloader.dataset.get_onehot_targets().to(args.device)
        train_unseen_mark_targets = train_unseen_mark_dataloader.dataset.get_onehot_targets().to(args.device)
        train_mark_targets = torch.cat((train_seen_mark_targets,train_unseen_mark_targets),dim=0)
        #生成  所有 已标注数据(原始已标+增量已标) 与 采样已标记数据(原始采样已标+增量采样已标) 的相似矩阵
        ALL_MARK_S = calc_similarity_matrix(ALL_MARK_TARGETS, train_mark_targets)


        # 从无标注数据集中采样
        train_seen_unmark_dataloader, sample_seen_unmark_omega,samples_seen_unmark_omega_index = sample_dataloader(
            seen_unmark_dataloader, args.num_samples, args.batch_size, args.root, args.dataset)
        train_unseen_unmark_dataloader, sample_unseen_unmark_omega,samples_unseen_unmark_omega_index = sample_dataloader(
            unseen_unmark_dataloader, args.num_samples*unmark_hp, args.batch_size*unmark_hp, args.root, args.dataset,all_hit_unseen_unmark_omega_index,args.sample_percent)

        train_seen_unmark_targets = train_seen_unmark_dataloader.dataset.get_onehot_targets().to(args.device)
        train_unseen_unmark_targets = train_unseen_unmark_dataloader.dataset.get_onehot_targets().to(args.device)

        # 用于DCC的量化
        UNSEEN_MARK_expand_U = torch.zeros(UNSEEN_MARK_B.shape).to(args.device)
        UNSEEN_UNMARK_expand_U = torch.zeros(UNSEEN_UNMARK_B.shape).to(args.device)
        temp_good_unseen_unmark_index_inOmega = []  # All_UNSEEN_UNMARK_DB 的 临时下标
        temp_good_unseen_unmark_index_inSample = []  # UNSEEN_UNMARK_Sample_U  的 临时下标

        # 使用创新型训练方式
        # 1.有标注数据
        total_mark_cnn_loss = 0
        itera = 0
        model.train()
        for batch, data in enumerate(zip(train_seen_mark_dataloader, train_unseen_mark_dataloader)):
            itera = itera + 1
            (data1, targets1, index1), (data2, targets2, index2) = data
            seen_mark_data, seen_mark_targets, seen_mark_index = data1.to(args.device), targets1.to(args.device), index1.to(args.device)
            unseen_mark_data, unseen_mark_targets, unseen_mark_index = data2.to(args.device), targets2.to(args.device), index2.to(args.device)

            seen_mark_u = model(seen_mark_data)
            SEEN_MARK_U[seen_mark_index] = seen_mark_u.data
            unseen_mark_u = model(unseen_mark_data)
            UNSEEN_MARK_U[unseen_mark_index] = unseen_mark_u.data

            temp_u = torch.cat((seen_mark_u,unseen_mark_u),dim=0)
            temp_targets = torch.cat((seen_mark_targets,unseen_mark_targets),dim=0)
            temp_S = torch.cat((ALL_MARK_S[seen_mark_index], ALL_MARK_S[(args.num_samples + unseen_mark_index)]), dim=0)
            ALL_MARK_B = torch.cat((SEEN_MARK_B, UNSEEN_MARK_B), dim=0)
            mark_cnn_loss = criterion(temp_u, ALL_MARK_B, temp_targets, temp_S, torch.ones(len(seen_mark_u)+len(unseen_mark_u)).to(args.device))

            optimizer.zero_grad()
            mark_cnn_loss.backward()
            optimizer.step()
            total_mark_cnn_loss = total_mark_cnn_loss + mark_cnn_loss.item()

        # update_database->B, 只优化增量的那部分数据，未增量的数据不优化，所以是num_seen_mark:
        unseen_mark_targets_index = get_unified_index_base_omega(all_unseen_mark_omega, torch.arange(0,args.num_samples*unmark_hp),sample_unseen_mark_omega)
        UNSEEN_MARK_expand_U[unseen_mark_targets_index, :] = UNSEEN_MARK_U
        # UNSEEN_MARK_B = update_B(UNSEEN_MARK_B, UNSEEN_MARK_U,ALL_MARK_S[args.num_samples:, len(all_seen_mark_targets):]
        #                          ,args.code_length, UNSEEN_MARK_expand_U, args.alpha)
        UNSEEN_MARK_B = batch_update_B(UNSEEN_MARK_B, ALL_MARK_S[args.num_samples:, len(all_seen_mark_targets):],
                                       UNSEEN_MARK_U, UNSEEN_MARK_expand_U, all_unseen_mark_targets,train_unseen_mark_targets, args)

        print('[iter:{}/{}][cnn_loss:{:.2f}][time:{:.2f}]'.format(
            it + 1, args.max_iter_unseen, total_mark_cnn_loss / itera, time.time() - iter_time))

        # 2.有标注数据+无标注数据
        # 训练无标注数据-》先生成伪标签，然后使用无标注数据训练模型
        all_good_acc = 0
        all_bad_acc = 0
        total_cnn_loss = 0
        itera2 = 0
        iter_time = time.time()
        for batch, data in enumerate(zip(train_seen_mark_dataloader,train_unseen_mark_dataloader, train_seen_unmark_dataloader,train_unseen_unmark_dataloader)):
            (data1, targets1, index1), (data2, targets2, index2),(data3, targets3, index3),(data4, targets4, index4) = data
            seen_mark_data, seen_mark_targets, seen_mark_index = data1.to(args.device), targets1.to(args.device), index1.to(args.device)
            unseen_mark_data, unseen_mark_targets, unseen_mark_index = data2.to(args.device), targets2.to(args.device), index2.to(args.device)
            seen_unmark_data, seen_unmark_targets, seen_unmark_index = data3.to(args.device), targets3.to(args.device), index3.to(args.device)
            unseen_unmark_data, unseen_unmark_targets, unseen_unmark_index = data4.to(args.device), targets4.to(args.device), index4.to(args.device)

            # 有标注数据
            seen_mark_u = model(seen_mark_data)
            SEEN_MARK_U[seen_mark_index] = seen_mark_u.data
            unseen_mark_u = model(unseen_mark_data)
            UNSEEN_MARK_U[unseen_mark_index] = unseen_mark_u.data
            # 无标注数据
            seen_unmark_u = model(seen_unmark_data)
            SEEN_UNMARK_U[seen_unmark_index] = seen_unmark_u.data
            unseen_unmark_u = model(unseen_unmark_data)
            UNSEEN_UNMARK_U[unseen_unmark_index] = unseen_unmark_u.data

            #获取还未生成伪标签的无标注数据
            doesnt_sample_index = get_doesnt_sample_index(all_hit_unseen_unmark_omega_index, all_unseen_unmark_omega,sample_unseen_unmark_omega, batch, int(args.batch_size)*unmark_hp)
            # 生成符合条件的伪标签(使用汉明距离，获得有标注数据库距离中心最远的点，如果无标注数据比 该点的还要接近于类中心，那么就判断为该标签)
            unseen_unmark_label,labels_prob, good_unseen_labels_index, bad_unseen_labels_index = get_unmark_label_byNeighbor(
                criterion.get_centerLoss().get_center_hash(), unseen_unmark_u, UNSEEN_MARK_B, all_unseen_mark_targets, args.classes_num,doesnt_sample_index, args.device)

            # 标签注入等操作-----------------------------------------------------------------
            # 没有命中的标签，不用于优化网络，但用于优化数据库，以及填充targets和B
            bad_unseen_unmark_label = unseen_unmark_label[bad_unseen_labels_index]
            #bad_unseen_unmark_index = unseen_unmark_index[bad_unseen_labels_index]  # (这里是 采样数据 的下标)
            #bad_unseen_unmark_u = unseen_unmark_u[bad_unseen_labels_index]
            bad_unseen_unmark_targets = unseen_unmark_targets[bad_unseen_labels_index]
            #bad_faker_targets_index = get_unified_index_base_omega(all_unseen_unmark_omega, bad_unseen_unmark_index, sample_unseen_unmark_omega)
            # 注入较差的伪标签，虽然差，但还是有利用价值的
            #all_unseen_unmark_facker_targets[bad_faker_targets_index] = bad_unseen_unmark_label.float()

            # 命中的标签, 用于优化网络，也用于优化数据库
            good_unseen_unmark_label = unseen_unmark_label[good_unseen_labels_index]
            good_unseen_unmark_index = unseen_unmark_index[good_unseen_labels_index]  # (这里是 采样数据 的下标)
            good_unseen_unmark_u = unseen_unmark_u[good_unseen_labels_index]
            good_unseen_unmark_targets = unseen_unmark_targets[good_unseen_labels_index]
            good_unseen_unmark_labels_prob = labels_prob[good_unseen_labels_index]
            # 根据数据库中omega一致性，定位出 命中标签 应该位于数据库数组(all_unseen_unmark_omega)的哪个索引下标  ,！！！！即当前batch 命中的全局下标（局部变量）
            good_faker_unseen_unmark_targets_index = get_unified_index_base_omega(all_unseen_unmark_omega, good_unseen_unmark_index,sample_unseen_unmark_omega)
            # 注入无标注数据的标签
            all_unseen_unmark_facker_targets[good_faker_unseen_unmark_targets_index] = good_unseen_unmark_label.float()
            # 记录 命中的索引到 全局Omega 的set集合中 ，因为每次采样可能会有重复数据，所以要set去重
            all_hit_unseen_unmark_omega_index = unrepeated_list(all_hit_unseen_unmark_omega_index, good_faker_unseen_unmark_targets_index)
            #用于DCC
            temp_good_unseen_unmark_index_inOmega.extend(good_faker_unseen_unmark_targets_index)  # 或者A+B  后面构建相似矩阵用到
            temp_good_unseen_unmark_index_inSample.extend(good_unseen_unmark_index.cpu().numpy())

            # 训练模型--------------------------------------------------------------------
            faker_seen_unmark_targets_index = get_unified_index_base_omega(all_seen_unmark_omega, seen_unmark_index, sample_seen_unmark_omega)
            all_targets = torch.cat((
                ALL_MARK_TARGETS,
                all_seen_unmark_facker_targets,
                all_unseen_unmark_facker_targets[all_hit_unseen_unmark_omega_index]
            ),dim=0)
            train_all_unmark = torch.cat((
                train_seen_mark_targets[seen_mark_index],
                train_unseen_mark_targets[unseen_mark_index],
                all_seen_unmark_facker_targets[faker_seen_unmark_targets_index],
                all_unseen_unmark_facker_targets[good_faker_unseen_unmark_targets_index],
            ),dim=0)
            temp_S = calc_similarity_matrix(all_targets,train_all_unmark)
            temp_u = torch.cat((seen_mark_u,unseen_mark_u,seen_unmark_u,good_unseen_unmark_u),dim=0)
            temp_b = torch.cat((SEEN_MARK_B,UNSEEN_MARK_B,SEEN_UNMARK_B, UNSEEN_UNMARK_B[all_hit_unseen_unmark_omega_index]), dim=0)

            # 归一到good_unseen_unmark_labels_prob.min()~1 的范围，即 a~1
            good_labels_prob = normalization_prob(good_unseen_unmark_labels_prob)
            prob = torch.cat((torch.ones(len(seen_mark_u)+len(unseen_mark_u)+len(seen_unmark_u)).to(args.device), good_labels_prob.to(args.device)), dim=0)

            # 有标注数据+命中的无标注数据  优化网络
            cnn_loss = criterion(temp_u, temp_b, train_all_unmark, temp_S,prob)
            optimizer.zero_grad()
            cnn_loss.backward()
            optimizer.step()
            total_cnn_loss = total_cnn_loss + cnn_loss.item()

            # calc accuracy
            good_acc = calc_discrimitor_acc(args, good_unseen_unmark_label, good_unseen_unmark_targets)
            bad_acc = calc_discrimitor_acc(args, bad_unseen_unmark_label, bad_unseen_unmark_targets)
            all_good_acc = all_good_acc + good_acc
            all_bad_acc = all_bad_acc + bad_acc
            itera2 = itera2 + 1

        # update_database->B----------------------------------------------------
        unseen_mark_targets_index = get_unified_index_base_omega(all_unseen_mark_omega, torch.arange(0,args.num_samples*unmark_hp),sample_unseen_mark_omega)
        UNSEEN_MARK_expand_U[unseen_mark_targets_index, :] = UNSEEN_MARK_U
        # UNSEEN_MARK_B = update_B(UNSEEN_MARK_B, UNSEEN_MARK_U,
        #          ALL_MARK_S[args.num_samples:, len(all_seen_mark_targets):],args.code_length, UNSEEN_MARK_expand_U, args.alpha)
        UNSEEN_MARK_B = batch_update_B(UNSEEN_MARK_B, ALL_MARK_S[args.num_samples:, len(all_seen_mark_targets):],
                                       UNSEEN_MARK_U, UNSEEN_MARK_expand_U, all_unseen_mark_targets,train_unseen_mark_targets, args)

        # 有标注数据 优化 all_good 优化命中的无标注数据
        UNSEEN_UNMARK_S = calc_similarity_matrix(
            all_unseen_unmark_facker_targets[all_hit_unseen_unmark_omega_index].double(),
            train_unseen_mark_targets #不需要train_seen_mark_targets，因为他们是不同的类别
        )
        # UNSEEN_UNMARK_B[all_hit_unseen_unmark_omega_index] = update_B(
        #     UNSEEN_UNMARK_B[all_hit_unseen_unmark_omega_index],
        #     UNSEEN_MARK_U, UNSEEN_UNMARK_S, args.code_length)
        UNSEEN_UNMARK_B[all_hit_unseen_unmark_omega_index] = batch_update_B(
            UNSEEN_UNMARK_B[all_hit_unseen_unmark_omega_index], UNSEEN_UNMARK_S,UNSEEN_MARK_U, None,
            all_unseen_unmark_facker_targets[all_hit_unseen_unmark_omega_index],train_unseen_mark_targets, args)

        logger.info('[iter:{}/{}][cnn_loss:{:.2f}][good_acc:{:.2f},bad_acc:{:.2f},fill_rate:{:.2f}/{:d}][time:{:.2f}]'.format(it + 1,
                                                                                                                 args.max_iter_unseen,
                                                                                                                 total_cnn_loss / itera2,
                                                                                                                 all_good_acc / itera2,
                                                                                                                 all_bad_acc / itera2,
                                                                                                                 len(all_hit_unseen_unmark_omega_index) / len(UNSEEN_UNMARK_B),
                                                                                                                 len(all_hit_unseen_unmark_omega_index),
                                                                                                                 time.time() - iter_time))
        # Evaluate----------------------------------------------------
        test_time = time.time()
        #增量后检索
        query_code, query_target = generate_code2(model, query_dataloader, args.code_length, args.classes_num,args.classes_num, args.device)
        #未增量检索
        query_code2, query_target2 = generate_code2(model, query_dataloader, args.code_length, args.classes_num,args.num_seen, args.device)
        ALL_MARK_B = torch.cat((SEEN_MARK_B,UNSEEN_MARK_B),dim=0)
        ALL_UNMARK_B = torch.cat((SEEN_UNMARK_B[all_hit_seen_unmark_omega_index], UNSEEN_UNMARK_B[all_hit_unseen_unmark_omega_index]), dim=0)
        ALL_B = torch.cat((ALL_MARK_B, ALL_UNMARK_B), dim=0)
        ALL_hit_UNMARK_TARGETS = torch.cat((all_seen_unmark_real_targets[all_hit_seen_unmark_omega_index], all_unseen_unmark_real_targets[all_hit_unseen_unmark_omega_index]), dim=0)
        ALL_DB_TARGETS = torch.cat((ALL_MARK_TARGETS, ALL_hit_UNMARK_TARGETS), dim=0)

        mAP = evaluate.mean_average_precision(query_code,ALL_MARK_B,query_target,ALL_MARK_TARGETS,args.device,topk=args.topk)
        mAP2 = evaluate.mean_average_precision(query_code,ALL_UNMARK_B,query_target,ALL_hit_UNMARK_TARGETS,args.device,topk=args.topk)
        all_mAP = evaluate.mean_average_precision(query_code,ALL_B,query_target,ALL_DB_TARGETS,args.device,topk=args.topk)

        ALL_SEEN_B = torch.cat((SEEN_MARK_B,SEEN_UNMARK_B[all_hit_seen_unmark_omega_index]),dim=0)
        ALL_SEEN_TARGETS = torch.cat((all_seen_mark_targets,all_seen_unmark_real_targets[all_hit_seen_unmark_omega_index]),dim=0)
        mAP3 = evaluate.mean_average_precision(query_code2,ALL_SEEN_B,query_target2,ALL_SEEN_TARGETS,args.device,topk=args.topk)

        prPath = os.path.join('iteratorData', args.version, "incre")
        # mAP = evaluate.calcMapAndPR(query_code, ALL_MARK_B, query_target, ALL_MARK_TARGETS,args.topk, prPath,str(it)+"-all_mark",True)
        # mAP2 = evaluate.calcMapAndPR(query_code, ALL_UNMARK_B, query_target, ALL_hit_UNMARK_TARGETS,args.topk, prPath, str(it)+"-all_unmark",True)
        # all_mAP = evaluate.calcMapAndPR(query_code, ALL_B, query_target, ALL_DB_TARGETS,args.topk, prPath, str(it)+"-all_mark_unmark",True)

        logger.info('[SSIDH-incre map1:{:.4f},map2:{:.4f},amap:{:.4f},or_map:{:.4f}][time:{:.2f}]'.format(mAP, mAP2,all_mAP,mAP3,time.time() - test_time))

        #保存散点图
        num=2000
        union_path = os.path.join('iteratorData', args.version, "incre")
        # #saveScatterImage(UNSEEN_MARK_U.cpu(), train_unseen_mark_targets.cpu(), num, union_path , str(it) + "-unseen_mark-U",args.classes_num-args.classes_num*args.num_seen/10)
        # #saveScatterImage(UNSEEN_MARK_U.cpu().sign(), train_unseen_mark_targets.cpu(), num,union_path , str(it) + "-unseen_mark-B",args.classes_num-args.classes_num*args.num_seen/10)
        # torch.save(UNSEEN_MARK_U.cpu(), union_path + "/" + str(it) + "-unseen_mark_u.t")
        # torch.save(train_unseen_mark_targets.int().cpu(), union_path + "/" + str(it) + "-unseen_mark_targets.t")
        #
        # #saveScatterImage(UNSEEN_UNMARK_U.cpu(), train_unseen_unmark_targets.cpu(), num, union_path , str(it) + "-unseen_unmark-U",args.classes_num-args.classes_num*args.num_seen/10)
        # #saveScatterImage(UNSEEN_UNMARK_U.cpu().sign(), train_unseen_unmark_targets.cpu(), num,union_path , str(it) + "-unseen_unmark-B",args.classes_num-args.classes_num*args.num_seen/10)
        # torch.save(UNSEEN_UNMARK_U.cpu(), union_path + "/" + str(it) + "-unseen_unmark_u.t")
        # torch.save(train_unseen_unmark_targets.int().cpu(), union_path + "/" + str(it) + "-unseen_unmark_targets.t")

        MARK_U = torch.cat((SEEN_MARK_U.cpu(),UNSEEN_MARK_U.cpu()),dim=0)
        UNMARK_U = torch.cat((SEEN_UNMARK_U.cpu(), UNSEEN_UNMARK_U.cpu()), dim=0)
        MARK_TARGET = torch.cat((train_seen_mark_targets.cpu(),train_unseen_mark_targets.cpu()),dim=0)
        UNMARK_TARGET = torch.cat((train_seen_unmark_targets.cpu(),train_unseen_unmark_targets.cpu()), dim=0)
        #saveScatterImage(MARK_U, MARK_TARGET, num,union_path , str(it) + "-mark-U",args.classes_num)
        #saveScatterImage(MARK_U.sign(), MARK_TARGET, num, union_path , str(it) + "-mark-B",args.classes_num)
        #saveScatterImage(UNMARK_U, UNMARK_TARGET, num, union_path , str(it) + "-unmark-U",args.classes_num)
        #saveScatterImage(UNMARK_U.sign(), UNMARK_TARGET, num, union_path , str(it) + "-unmark-B",args.classes_num)

        ALL_U = torch.cat((MARK_U,UNMARK_U),dim=0)
        ALL_TARGETS = torch.cat((MARK_TARGET, UNMARK_TARGET), dim=0)
        #saveScatterImage(ALL_U, ALL_TARGETS, num+1000, union_path , str(it) + "-ALL_U",args.classes_num)
        #saveScatterImage(ALL_U.sign(), ALL_TARGETS, num+1000, union_path , str(it) + "-ALL_B",args.classes_num)
        torch.save(ALL_U.cpu(), union_path + "/" + str(it) + "-all_u.t")
        torch.save(ALL_TARGETS.int().cpu(), union_path + "/" + str(it) + "-all_targets.t")

        # 保存模型 信息
        union_path2 = os.path.join('iteratorData', args.version,"incre","final")
        torch.save(UNSEEN_MARK_B.cpu(), os.path.join(union_path2, 'unseen_mark_b.t'))
        torch.save(all_unseen_mark_targets.int().cpu(),os.path.join(union_path2, 'unseen_mark_targets.t'))
        torch.save(UNSEEN_UNMARK_B.cpu(), os.path.join(union_path2, 'unseen_unmark_b.t'))
        torch.save(all_unseen_unmark_real_targets.int().cpu(), os.path.join(union_path2, 'unseen_unmark_real_targets.t'))
        torch.save(all_unseen_unmark_facker_targets.int().cpu(),os.path.join(union_path2, 'unseen_unmark_facker_targets.t'))

        torch.save(all_hit_unseen_unmark_omega_index, os.path.join(union_path2, 'all_hit_unseen_unmark_omega_index.t'))
        torch.save(model.state_dict(), os.path.join(union_path2, 'model.pt'))

    logger.info('Training SSIDH-base finish, time:{:.2f}'.format(time.time() - total_time))

