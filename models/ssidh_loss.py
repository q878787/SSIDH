import torch.nn as nn
import torch

class SSIDH_Loss(nn.Module):
    """
    Loss function of SSIDH.
    Args:
        class_num(int):classed count
        code_length(int): Hashing code length.
        gamma, alpha(float): Hyper-parameter.
    """
    def __init__(self, center_loss, code_length, alpha):
        super(SSIDH_Loss, self).__init__()
        self.CenterLoss = center_loss
        self.code_length = code_length
        self.alpha = alpha


    def forward(self, U, B, Y, S,prob):
        #U = self.htanh.apply(U,self.CenterLoss.label2center(Y))
        center_loss = (self.CenterLoss(U, Y)*prob).mean()
        keep_loss = (((self.code_length * S - U @ B.t()) ** 2).mean(dim=1)*prob).mean()
        loss = keep_loss + self.alpha * center_loss  # + quantization_loss#
        return loss

    def get_centerLoss(self):
        return self.CenterLoss



# class SSIDH_Loss(nn.Module):
#     """
#     Loss function of SSIDH.
#     Args:
#         class_num(int):classed count
#         code_length(int): Hashing code length.
#         gamma, alpha(float): Hyper-parameter.
#     """
#     def __init__(self, center_loss, code_length, alpha):
#         super(SSIDH_Loss, self).__init__()
#         self.CenterLoss = center_loss
#         self.code_length = code_length
#         self.alpha = alpha
#
#     def forward(self, U, B, Y, S,prob,bad_unmark_u,bad_unmark_label,bad_labels_prob):
#         '''
#
#         :param U: good U
#         :param B: all good DB
#         :param Y: good Y
#         :param S: good similarly martix
#         :param prob: good prob
#         :param bad_unmark_u: bad U
#         :param bad_unmark_label: bad target
#         :param bad_labels_prob: bad prob
#         :return: loss
#         '''
#         #U = SSIDH_Loss.Htanh.apply(U)
#         #U = U.tanh()
#         center_loss = (self.CenterLoss(U, Y).mean(dim=1)*prob).mean()
#         # if bad_unmark_u != None:
#         #     #b1-b2(0~2)->(b1-b2)**2(0~4),希望他们越大loss才会越小，所以要4-(b1-b2)**2
#         #     bad_center_loss = ((4-self.CenterLoss(bad_unmark_u, bad_unmark_label)).mean(dim=1)*(1-bad_labels_prob)).mean()
#         #     center_loss = (center_loss+bad_center_loss/len(bad_unmark_label[0]))/2
#
#         keep_loss = (((self.code_length * S - U @ B.t()) ** 2).mean(dim=1)*prob).mean()
#         loss = keep_loss + self.alpha * center_loss
#         return loss
#
#     def get_centerLoss(self):
#         return self.CenterLoss
#

