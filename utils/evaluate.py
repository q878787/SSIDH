import torch
from tqdm import tqdm
import numpy as np
import json
import os

def mean_average_precision(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=None,
                           ):
    """
    Calculate mean average precision(map).
    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.
    Returns:
        meanAP (float): Mean Average Precision.
    """
    query_code = query_code.to(device)
    database_code = database_code.to(device)
    query_labels = query_labels.to(device)
    database_labels = database_labels.to(device)

    if topk > len(database_code):
        topk = len(database_code)-1

    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP


#使用了tie方法检测，准确度会变高
def mean_average_precision2(query_code,
                            database_code,
                            query_labels,
                            database_labels,
                            device=None,
                            topk=1000,
                            ):
    """
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.

    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = query_labels.shape[0]
    mean_AP = torch.zeros(1).cuda()
    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance

        hamming_index = torch.argsort(hamming_dist)
        retrieval = retrieval[hamming_index][:topk]
        #
        hamming_order = hamming_dist[hamming_index][:topk]

        k = 0
        retrieval = retrieval.cpu().numpy()
        hamming_order = hamming_order.cpu().numpy()
        last_dis = hamming_order[0]

        #         start=time.time()
        for j in range(0, topk):
            if last_dis != hamming_order[j]:
                k = j
                last_dis = hamming_order[j]

            if retrieval[j] == 1 and k != j:
                retrieval[k] = 1
                k += 1
                retrieval[j] = 0
        #         print(time.time()-start)
        retrieval = torch.from_numpy(retrieval).cuda()
        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).cuda()

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return float(mean_AP)


def calcMapAndPR(tst_binary,trn_binary,tst_label,trn_label,topK,savePath,filename,isContainPR=False):
    tst_binary = tst_binary.cpu()
    trn_binary = trn_binary.cpu()
    tst_label = tst_label.cpu()
    trn_label = trn_label.cpu()
    num_dataset = len(trn_binary)

    #判断topK是否大于DB数
    if topK>len(trn_binary):
        topK = len(trn_binary)-1

    if isContainPR:
        mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), topK)
        return mAP

    # need more memory
    mAP, cum_prec, cum_recall = CalcTopMapWithPR(tst_binary.numpy(), tst_label.numpy(),
                                                 trn_binary.numpy(), trn_label.numpy(),
                                                 topK)
    index_range = num_dataset // 100
    index = [i * 100 - 1 for i in range(1, index_range + 1)]
    max_index = max(index)
    overflow = num_dataset - index_range * 100
    index = index + [max_index + i for i in range(1, overflow + 1)]
    c_prec = cum_prec[index]
    c_recall = cum_recall[index]

    pr_data = {
        "index": index,
        "P": c_prec.tolist(),
        "R": c_recall.tolist()
    }
    os.makedirs(savePath, exist_ok=True)
    with open(savePath+"/"+ filename+f"_%d" % (mAP * 1000) + ".json",'w') as f:
        f.write(json.dumps(pr_data))

    return mAP

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

# faster but more memory
def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap