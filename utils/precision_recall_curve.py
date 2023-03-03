import matplotlib.pyplot as plt
import json
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
# Precision Recall Curve data

#cifar-10
pr_data = {
    "DPSH": "./pr/cifar10/[DPSH]_cifar10_744.json",
    "HashNet": "./pr/cifar10/[HashNet]_cifar10_757.json",
    "DTSH": "./pr/cifar10/[DTSH]_cifar10_765.json",
    "DCH": "./pr/cifar10/[DCH]_cifar10_790.json",
    "GAH": "./pr/cifar10/[GAH]_cifar10_753.json",
    "CSQ": "./pr/cifar10/[CSQ]_cifar10_801.json",
    "ADSH": "./pr/cifar10/[ADSH]_cifar10_906.json",
    "DIHN": "./pr/cifar10/[DIHN]_cifar10_901.json",
    "SIDH": "./pr/cifar10/[SIDH]_cifar10_922.json",
    #"SSIDH": "./pr/cifar10/[SSIDH]_cifar10_902.json",
    "SIDH+SQG": "./pr/cifar10/[SIDH+SQG]_cifar10_935.json",
}
save_path = "Cifar10"
N1 = 50
y1 = 0.1
y11 = 1.02
N2 = 100
y2 =0.4
y22 =1.0
N3 = 70
y3 = 0.5
y33 = 1.0


pr_data = {
    "DPSH": "./pr/imagenet/[DPSH]_imagenet_473.json",
    "HashNet": "./pr/imagenet/[HashNet]_imagenet_630.json",#xx
    "DTSH": "./pr/imagenet/[DTSH]_imagenet_556.json",
    "GAH": "./pr/imagenet/[GAH]_imagenet_623.json",
    "DCH": "./pr/imagenet/[DCH]_imagenet_622.json",
    "CSQ": "./pr/imagenet/[CSQ]_imagenet_653.json",
    "ADSH": "./pr/imagenet/[ADSH]_imagenet_728.json",
    "DIHN": "./pr/imagenet/[DIHN]_imagenet_754.json",
    "SIDH": "./pr/imagenet/[SIDH]_imagenet_774.json",
    "SSIDH": "./pr/imagenet/[SSIDH]_imagenet_740.json",
    #"SIDH+SQG": "./pr/imagenet/[SIDH+SQG]_imagenet_780.json",
}
save_path = "Imagenet100"
N1 = 70
y1 = 0.0
y11 = 0.9
N2 = 30
y2 =0.2
y22 =0.9
N3 = 30
y3 = 0.2
y33 = 0.9
#
#nuswide
# pr_data = {
#     "DHN": "../prImg/nuswide/ada/[DHN]_nuswide_21_758.json",
#     "DPSH": "../prImg/nuswide/ada/[DPSH]_nuswide_21_809.json",
#     "HashNet": "../prImg/nuswide/ada/[HashNet]_nuswide_21_756.json",
#     "DTSH": "../prImg/nuswide/ada/[DTSH]_nuswide_21_819.json",
#     "DCH": "../prImg/nuswide/ada/[DCH]_nuswide_21_711.json",
#     "GAH": "../prImg/nuswide/ada/[GAH]_nuswide_21_767.json",
#     "DMMH": "../prImg/nuswide/ada/[DMMH]_nuswide_21_732.json",
#     "DAMH": "../prImg/nuswide/ada/[DAMH]_nuswide_21_824.json",
# }
# save_path = "nuswide"
# N1 = 100
# y1 = 0.36
# y11 = 0.85
# N2 = 300
# y2 =0.65
# y22 =0.85
# N3 = 100
# y3 = 0.65
# y33 = 0.85

#plt.style.use("seaborn-darkgrid")
N = 100
lwid = 3
# N = -1
for key in pr_data:
    path = pr_data[key]
    pr_data[key] = json.load(open(path))


# markers = "DdsPvo*xH1234h"
markers = ".........................."
method2marker = {}
i = 0
for method in pr_data:
    method2marker[method] = markers[i]
    i += 1

from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import make_interp_spline
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False     # 正常显示负号
plt.rcParams.update({'font.size': 20})


plt.figure(figsize=(24, 8))
plt.subplots_adjust(wspace=0.26, hspace=0,left=0.05,bottom=0.09,right=0.995,top=0.920)
#pr
plt.subplot(131)
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"],pr_data[method]["R"],pr_data[method]["index"]
    print(len(P))
    print(len(R))
    # R = np.array(R)
    # P = np.array(P)
    # R, ind = np.unique(R, return_index=True)
    # P = P[ind]
    # model = make_interp_spline(R, P)
    # xs = np.linspace(0, 1, 1000)
    # ys = model(xs)
    # plt.plot(xs, ys,linewidth=lwid)
    #plt.plot(R, P, linestyle='solid', marker=method2marker[method], label=method)
    plt.plot(R, P, linestyle='-', linewidth=lwid, label=method)

plt.grid(True)
plt.xlim(0, 1)
plt.ylim(y1, y11)
plt.xlabel('recall',fontsize=18)
plt.ylabel('precision',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(prop={'size': 20})

#recall
plt.subplot(132)
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N2],pr_data[method]["R"][:N2],pr_data[method]["index"][:N2]

    plt.plot(draw_range, P, linestyle='-', linewidth=lwid, label=method)
    #plt.plot(draw_range, R, linestyle="-", marker=method2marker[method],linewidth=lwid, label=method)
plt.xlim(0, max(draw_range))
plt.ylim(y2, y22)
plt.grid(True)
plt.xlabel('The number of retrieved samples',fontsize=18)
plt.ylabel('recall',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.legend()

#precision
plt.subplot(133)
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N3],pr_data[method]["R"][:N3],pr_data[method]["index"][:N3]
    plt.plot(draw_range, P, linestyle='-', linewidth=lwid, label=method)
    #plt.plot(draw_range, P, linestyle="-", marker=method2marker[method],linewidth=lwid, label=method)
plt.xlim(0, max(draw_range))
plt.ylim(y3, y33)
plt.grid(True)
plt.xlabel('The number of retrieved samples',fontsize=18)
plt.ylabel('precision',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.legend()

plt.suptitle(save_path,fontsize=25)
plt.savefig(save_path + '.png')
plt.savefig(save_path + '.pdf')
plt.savefig(save_path + '.svg')

plt.show()
