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
                     args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr / 10, )

    # 构建该数据集包含 标注和未标注数据的   完整数据库 B  ，以及每次采样的数据 U
    SEEN_MARK_B = torch.randn(len(seen_mark_dataloader.dataset), args.code_length).sign().to(args.device)
    SEEN_MARK_U = torch.zeros(args.num_samples, args.code_length).to(args.device)
    all_seen_mark_targets = seen_mark_dataloader.dataset.get_onehot_targets().to(args.device)
    all_seen_mark_omega = seen_mark_dataloader.dataset.get_omega().to(args.device)

    SEEN_UNMARK_B = torch.randn(len(seen_unmark_dataloader.dataset), args.code_length).sign().to(args.device)
    SEEN_UNMARK_U = torch.zeros(args.num_samples, args.code_length).to(args.device)  # 采样样本的临时数据库
    all_seen_unmark_real_targets = seen_unmark_dataloader.dataset.get_onehot_targets().to(args.device)  # 真实标签
    all_seen_unmark_facker_targets = (torch.zeros(len(seen_unmark_dataloader.dataset), args.classes_num)).to(args.device)  # 伪标签
    all_seen_unmark_omega = seen_unmark_dataloader.dataset.get_omega().to(args.device)  # 完整数据集中 生成的无标注数据 下标
    # 这个数组记录命中的标签 在数据库中的下标 ，其中用了set去重
    all_hit_seen_unmark_omega_index = []

    total_time = time.time()
    for it in range(args.max_iter_seen):
        iter_time = time.time()

        SEEN_MARK_B = ((SEEN_MARK_B + torch.randn(len(seen_mark_dataloader.dataset), args.code_length).to(args.device))).sign()
        SEEN_UNMARK_B[all_hit_seen_unmark_omega_index] = ((SEEN_UNMARK_B[all_hit_seen_unmark_omega_index] + torch.randn(len(all_hit_seen_unmark_omega_index), args.code_length).to(args.device))).sign()

        # 从有标注数据的训练集中采样
        train_seen_mark_dataloader, sample_seen_mark_omega,samples_seen_mark_omega_index = sample_dataloader(
            seen_mark_dataloader, args.num_samples, args.batch_size, args.root, args.dataset)
        # 标注数据的生成相似矩阵
        train_seen_mark_targets = train_seen_mark_dataloader.dataset.get_onehot_targets().to(args.device)
        SEEN_MARK_S = calc_similarity_matrix(all_seen_mark_targets, train_seen_mark_targets)

        # 从无标注数据集中采样
        train_seen_unmark_dataloader, sample_seen_unmark_omega,samples_seen_unmark_omega_index = sample_dataloader(
            seen_unmark_dataloader, args.num_samples, args.batch_size, args.root, args.dataset,
            all_hit_seen_unmark_omega_index, args.sample_percent)

        train_seen_unmark_targets = train_seen_unmark_dataloader.dataset.get_onehot_targets().to(args.device)

        # 用于DCC的量化
        SEEN_MARK_expand_U = torch.zeros(SEEN_MARK_B.shape).to(args.device)
        SEEN_UNMARK_expand_U = torch.zeros(SEEN_UNMARK_B.shape).to(args.device)
        temp_good_seen_unmark_index_inOmega = []  # 基于 All_UNMARK_DB 的 临时下标
        temp_good_seen_unmark_index_inSample = []  # 基于 Sample_U  的 临时下标

        # 使用创新型训练方式
        # 1.有标注数据
        total_mark_cnn_loss = 0
        itera = 0
        model.train()
        for batch, (data, targets, index) in enumerate(train_seen_mark_dataloader):
            itera = itera + 1
            mark_data, mark_targets, mark_index = data.to(args.device), targets.to(args.device), index.to(args.device)
            mark_u = model(mark_data)
            SEEN_MARK_U[mark_index] = mark_u.data

            mark_cnn_loss = criterion(mark_u, SEEN_MARK_B, mark_targets, SEEN_MARK_S[mark_index],torch.ones(len(mark_u)).to(args.device))
            optimizer.zero_grad()
            mark_cnn_loss.backward()
            optimizer.step()
            total_mark_cnn_loss = total_mark_cnn_loss + mark_cnn_loss.item()

        #update_database->B----------------------------------------------------
        mark_targets_index = get_unified_index_base_omega(all_seen_mark_omega, torch.arange(0, args.num_samples),sample_seen_mark_omega)
        SEEN_MARK_expand_U[mark_targets_index, :] = SEEN_MARK_U
        # SEEN_MARK_B = update_B(SEEN_MARK_B, SEEN_MARK_U, SEEN_MARK_S, args.code_length, SEEN_MARK_expand_U, args.alpha)
        SEEN_MARK_B = batch_update_B(SEEN_MARK_B, SEEN_MARK_S, SEEN_MARK_U, SEEN_MARK_expand_U, all_seen_mark_targets,train_seen_mark_targets, args)

        print('[iter:{}/{}][cnn_loss:{:.2f}][time:{:.2f}]'.format(
            it + 1, args.max_iter_seen, total_mark_cnn_loss / itera, time.time() - iter_time))

        # 2.有标注数据+无标注数据
        # 训练无标注数据-》先生成伪标签，然后使用无标注数据训练模型
        all_good_acc = 0
        all_bad_acc = 0
        total_cnn_loss = 0
        itera = 0
        iter_time = time.time()
        for batch, data in enumerate(zip(train_seen_mark_dataloader, train_seen_unmark_dataloader)):
            (data1, targets1, index1), (data2, targets2, index2) = data
            mark_data, mark_targets, mark_index = data1.to(args.device), targets1.to(args.device), index1.to(args.device)
            unmark_data, unmark_targets, unmark_index = data2.to(args.device), targets2.to(args.device), index2.to(args.device)

            mark_u = model(mark_data)
            SEEN_MARK_U[mark_index] = mark_u.data
            unmark_u = model(unmark_data)
            SEEN_UNMARK_U[unmark_index] = unmark_u.data

            #获取还未生成伪标签的无标注数据
            doesnt_sample_index = get_doesnt_sample_index(all_hit_seen_unmark_omega_index, all_seen_unmark_omega,sample_seen_unmark_omega,batch, int(args.batch_size))
            # 生成符合条件的伪标签(使用汉明距离，获得有标注数据库距离中心最远的点，如果无标注数据比 该点的还要接近于类中心，那么就判断为该标签)
            unmark_label, labels_prob, good_labels_index, bad_labels_index = get_unmark_label_byNeighbor(
                criterion.get_centerLoss().get_center_hash(), unmark_u, SEEN_MARK_B, all_seen_mark_targets,
                args.classes_num,doesnt_sample_index, args.device)

            # 标签注入等操作-----------------------------------------------------------------
            # 没有命中的标签，不用于优化网络，但用于优化数据库，以及填充targets和B
            bad_unmark_label = unmark_label[bad_labels_index]
            # bad_unmark_index = unmark_index[bad_labels_index]  # (这里是 采样数据 的下标)
            # bad_unmark_u = unmark_u[bad_labels_index]
            bad_unmark_targets = unmark_targets[bad_labels_index]
            # bad_faker_targets_index = get_unified_index_base_omega(all_seen_unmark_omega, bad_unmark_index,sample_seen_unmark_omega)
            # 注入较差的伪标签，虽然差，但还是有利用价值的
            # all_seen_unmark_facker_targets[bad_faker_targets_index] = bad_unmark_label.float()

            # 命中的标签 以及 其距离类中心的相似概率, 用于优化网络，也用于优化数据库
            good_unmark_label = unmark_label[good_labels_index]
            good_unmark_index = unmark_index[good_labels_index]  # (这里是 采样数据 的下标)
            good_unmark_u = unmark_u[good_labels_index]
            good_unmark_targets = unmark_targets[good_labels_index]
            good_labels_prob = labels_prob[good_labels_index]
            # 根据数据库中omega一致性，定位出 命中标签 应该位于数据库数组(all_seen_unmark_omega)的哪个索引下标
            good_faker_targets_index = get_unified_index_base_omega(all_seen_unmark_omega, good_unmark_index,sample_seen_unmark_omega)
            # 注入无标注数据的标签
            all_seen_unmark_facker_targets[good_faker_targets_index] = good_unmark_label.float()
            # 记录 命中的索引到 全局Omega 的set集合中 ，因为每次采样可能会有重复数据，所以要set去重
            all_hit_seen_unmark_omega_index = unrepeated_list(all_hit_seen_unmark_omega_index, good_faker_targets_index)
            # 用于DCC
            temp_good_seen_unmark_index_inOmega.extend(good_faker_targets_index)  # 或者A+B  后面构建相似矩阵用到
            temp_good_seen_unmark_index_inSample.extend(good_unmark_index.cpu().numpy())

            # 训练模型--------------------------------------------------------------------
            temp_S = calc_similarity_matrix(
                torch.cat((all_seen_mark_targets, all_seen_unmark_facker_targets[all_hit_seen_unmark_omega_index]),dim=0),
                torch.cat((train_seen_mark_targets[mark_index], all_seen_unmark_facker_targets[good_faker_targets_index]),dim=0)
            )
            u = torch.cat((mark_u, good_unmark_u), dim=0)
            b = torch.cat((SEEN_MARK_B, SEEN_UNMARK_B[all_hit_seen_unmark_omega_index]), dim=0)
            temp_targets = torch.cat((mark_targets, good_unmark_label), dim=0)

            # 归一到good_labels_prob.min()~1 的范围，即 a~1
            good_labels_prob = normalization_prob(good_labels_prob)
            prob = torch.cat((torch.ones(len(mark_u)).to(args.device), good_labels_prob.to(args.device)), dim=0)

            # 有标注数据+命中的无标注数据  优化网络
            cnn_loss = criterion(u, b, temp_targets, temp_S, prob)
            optimizer.zero_grad()
            cnn_loss.backward()
            optimizer.step()
            total_cnn_loss = total_cnn_loss + cnn_loss.item()

            # calc accuracy
            good_acc = calc_discrimitor_acc(args, good_unmark_label, good_unmark_targets)
            bad_acc = calc_discrimitor_acc(args, bad_unmark_label, bad_unmark_targets)
            all_good_acc = all_good_acc + good_acc
            all_bad_acc = all_bad_acc + bad_acc
            itera = itera + 1

        # update_database->B----------------------------------------------------
        # 有标注数据 优化 整个 有标注数据库
        mark_targets_index = get_unified_index_base_omega(all_seen_mark_omega, torch.arange(0, args.num_samples),sample_seen_mark_omega)
        SEEN_MARK_expand_U[mark_targets_index, :] = SEEN_MARK_U
        #SEEN_MARK_B = update_B(SEEN_MARK_B, SEEN_MARK_U, SEEN_MARK_S, args.code_length, SEEN_MARK_expand_U, args.alpha)
        SEEN_MARK_B = batch_update_B(SEEN_MARK_B, SEEN_MARK_S, SEEN_MARK_U, SEEN_MARK_expand_U, all_seen_mark_targets,train_seen_mark_targets, args)

        # 有标注数据 优化 all_good 无标注数据库
        SEEN_UNMARK_S = calc_similarity_matrix(
            all_seen_unmark_facker_targets[all_hit_seen_unmark_omega_index],
            train_seen_mark_targets.float()
        )
        # SEEN_UNMARK_B[all_hit_seen_unmark_omega_index] = update_B(
        #     SEEN_UNMARK_B[all_hit_seen_unmark_omega_index], SEEN_MARK_U, SEEN_UNMARK_S, args.code_length)
        SEEN_UNMARK_B[all_hit_seen_unmark_omega_index] = batch_update_B(SEEN_UNMARK_B[all_hit_seen_unmark_omega_index],
            SEEN_UNMARK_S, SEEN_MARK_U, None, all_seen_unmark_facker_targets[all_hit_seen_unmark_omega_index],train_seen_mark_targets, args)

        # end training----------------------------------------------------
        logger.info('[iter:{}/{}][loss:{:.2f}][good_acc:{:.2f},bad_acc:{:.2f},fill_rate:{:.2f}/{:d}][time:{:.2f}]'.format(
                it + 1,
                args.max_iter_seen,
                total_cnn_loss / itera,
                all_good_acc / itera,
                all_bad_acc / itera,
                len(all_hit_seen_unmark_omega_index) / len(SEEN_UNMARK_B),
                len(all_hit_seen_unmark_omega_index),
                time.time() - iter_time))
        # Evaluate----------------------------------------------------
        test_time = time.time()
        query_code, query_target = generate_code2(model, query_dataloader, args.code_length, args.classes_num,args.num_seen, args.device)
        ALL_SEEN_B = torch.cat((SEEN_MARK_B, SEEN_UNMARK_B[all_hit_seen_unmark_omega_index]), dim=0)
        ALL_SEEN_TARGET = torch.cat((all_seen_mark_targets, all_seen_unmark_real_targets[all_hit_seen_unmark_omega_index]), dim=0)

        mAP = evaluate.mean_average_precision(query_code,SEEN_MARK_B,query_target,all_seen_mark_targets,args.device,topk=args.topk)
        mAP2 = evaluate.mean_average_precision(query_code,SEEN_UNMARK_B[all_hit_seen_unmark_omega_index],query_target,all_seen_unmark_real_targets[all_hit_seen_unmark_omega_index],args.device,topk=args.topk)
        all_mAP = evaluate.mean_average_precision(query_code,ALL_SEEN_B,query_target,ALL_SEEN_TARGET,args.device,topk=args.topk)

        prPath = os.path.join('iteratorData', args.version, "basic")
        # mAP = evaluate.calcMapAndPR(query_code, SEEN_MARK_B, query_target, all_seen_mark_targets, args.topk, prPath,str(it) + "-seen_mark", True)
        # mAP2 = evaluate.calcMapAndPR(query_code, SEEN_UNMARK_B[all_hit_seen_unmark_omega_index], query_target,
        #                              all_seen_unmark_real_targets[all_hit_seen_unmark_omega_index],
        #                              args.topk, prPath, str(it) + "-senn_unmark", True)
        # all_mAP = evaluate.calcMapAndPR(query_code, ALL_SEEN_B, query_target, ALL_SEEN_TARGET, args.topk, prPath,str(it) + "-all_seen_mark_unmark", True)

        logger.info('[SSIDH-base map1:{:.4f},map2:{:.4f},amap:{:.4f}][time:{:.2f}]'.format(mAP, mAP2, all_mAP,time.time() - test_time))

        # 保存散点图
        num = 2000
        union_path = os.path.join('iteratorData', args.version, "basic")
        # #saveScatterImage(SEEN_MARK_U.cpu(), train_seen_mark_targets.cpu(), num, union_path , str(it) + "-seen_mark-U",args.classes_num*args.num_seen/10)
        # saveScatterImage(SEEN_MARK_U.cpu().sign(), train_seen_mark_targets.cpu(), num,union_path , str(it) + "-seen_mark-B",args.classes_num*args.num_seen/10)
        # torch.save(SEEN_MARK_U.cpu(), union_path + "/" + str(it) + "-seen_mark_u.t")
        # torch.save(train_seen_mark_targets.int().cpu(), union_path + "/" + str(it) + "-seen_mark_targets.t")
        #
        # #saveScatterImage(SEEN_UNMARK_U.cpu(), train_seen_unmark_targets.cpu(), num, union_path , str(it) + "-seen_unmark-U",args.classes_num*args.num_seen/10)
        # saveScatterImage(SEEN_UNMARK_U.cpu().sign(), train_seen_unmark_targets.cpu(), num,union_path , str(it) + "-seen_unmark-B",args.classes_num*args.num_seen/10)
        # torch.save(SEEN_UNMARK_U.cpu(), union_path + "/" + str(it) + "-seen_unmark_u.t")
        # torch.save(train_seen_unmark_targets.int().cpu(), union_path + "/" + str(it) + "-seen_unmark_targets.t")

        SEEN_TRAIN_U = torch.cat((SEEN_MARK_U, SEEN_UNMARK_U), dim=0)
        SEEN_TRAIN_TARGET = torch.cat((train_seen_mark_targets, train_seen_unmark_targets), dim=0)
        # saveScatterImage(SEEN_TRAIN_U.cpu(), SEEN_TRAIN_TARGET.cpu(), num, union_path, str(it) + "-seen_all-U", args.classes_num * args.num_seen / 10)
        #saveScatterImage(SEEN_TRAIN_U.cpu().sign(), SEEN_TRAIN_TARGET.cpu(), num, union_path, str(it) + "-seen_all-B",args.classes_num * args.num_seen / 10)
        torch.save(SEEN_TRAIN_U.cpu(), union_path + "/" + str(it) + "-seen_all_u.t")
        torch.save(SEEN_TRAIN_TARGET.int().cpu(), union_path + "/" + str(it) + "-seen_all_targets.t")

        # 保存模型
        union_path2 = os.path.join('iteratorData', args.version, "basic", "final")
        torch.save(SEEN_MARK_B.cpu(), os.path.join(union_path2, 'seen_mark_b.t'))
        torch.save(all_seen_mark_targets.int().cpu(), os.path.join(union_path2, 'all_seen_mark_targets.t'))
        torch.save(SEEN_UNMARK_B.cpu(), os.path.join(union_path2, 'seen_unmark_b.t'))
        torch.save(all_seen_unmark_real_targets.int().cpu(),os.path.join(union_path2, 'all_seen_unmark_real_targets.t'))
        torch.save(all_seen_unmark_facker_targets.int().cpu(),os.path.join(union_path2, 'all_seen_unmark_facker_targets.t'))

        torch.save(all_hit_seen_unmark_omega_index, os.path.join(union_path2, 'all_hit_seen_unmark_omega_index.t'))
        torch.save(model.state_dict(), os.path.join(union_path2, 'model.pt'))

    logger.info('Training SSIDH-base finish, time:{:.2f}'.format(time.time() - total_time))
    return SEEN_MARK_B, SEEN_UNMARK_B, all_seen_unmark_facker_targets, all_hit_seen_unmark_omega_index




