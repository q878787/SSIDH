import os
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")


def saveScatterImage(x, y, num, save_path, fileName, class_num=10):
    # if class_num > 10:
    #     class_num = 100
    os.makedirs(save_path, exist_ok=True)
    # 打乱他
    index11 = np.random.permutation(len(x))
    x = x[index11]
    y = y[index11].int()

    x = x[:num].cpu()
    y = torch.topk(y, 1)[1].squeeze(1)[:num].cpu()
    tsne = manifold.TSNE(n_components=2, learning_rate='auto', init='pca', random_state=2022)
    X_tsne = tsne.fit_transform(x)
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(wspace=0.2, hspace=0.2, left=0.06, bottom=0.04, right=0.99, top=0.99)

    colors = {'aliceblue': '#F0F8FF', 'antiquewhite': '#FAEBD7', 'aqua': '#00FFFF', 'aquamarine': '#7FFFD4',
              'azure': '#F0FFFF', 'beige': '#F5F5DC', 'bisque': '#FFE4C4', 'black': '#000000',
              'blanchedalmond': '#FFEBCD', 'blue': '#0000FF', 'blueviolet': '#8A2BE2', 'brown': '#A52A2A',
              'burlywood': '#DEB887', 'cadetblue': '#5F9EA0', 'chartreuse': '#7FFF00', 'chocolate': '#D2691E',
              'coral': '#FF7F50', 'cornflowerblue': '#6495ED', 'cornsilk': '#FFF8DC', 'crimson': '#DC143C',
              'cyan': '#00FFFF', 'darkblue': '#00008B', 'darkcyan': '#008B8B', 'darkgoldenrod': '#B8860B',
              'darkgray': '#A9A9A9', 'darkgreen': '#006400', 'darkkhaki': '#BDB76B', 'darkmagenta': '#8B008B',
              'darkolivegreen': '#556B2F', 'darkorange': '#FF8C00', 'darkorchid': '#9932CC', 'darkred': '#8B0000',
              'darksalmon': '#E9967A', 'darkseagreen': '#8FBC8F', 'darkslateblue': '#483D8B',
              'darkslategray': '#2F4F4F', 'darkturquoise': '#00CED1', 'darkviolet': '#9400D3', 'deeppink': '#FF1493',
              'deepskyblue': '#00BFFF', 'dimgray': '#696969', 'dodgerblue': '#1E90FF', 'firebrick': '#B22222',
              'floralwhite': '#FFFAF0', 'forestgreen': '#228B22', 'fuchsia': '#FF00FF', 'gainsboro': '#DCDCDC',
              'ghostwhite': '#F8F8FF', 'gold': '#FFD700', 'goldenrod': '#DAA520', 'gray': '#808080', 'green': '#008000',
              'greenyellow': '#ADFF2F', 'honeydew': '#F0FFF0', 'hotpink': '#FF69B4', 'indianred': '#CD5C5C',
              'indigo': '#4B0082', 'ivory': '#FFFFF0', 'khaki': '#F0E68C', 'lavender': '#E6E6FA',
              'lavenderblush': '#FFF0F5', 'lawngreen': '#7CFC00', 'lemonchiffon': '#FFFACD', 'lightblue': '#ADD8E6',
              'lightcoral': '#F08080', 'lightcyan': '#E0FFFF', 'lightgoldenrodyellow': '#FAFAD2',
              'lightgreen': '#90EE90', 'lightgray': '#D3D3D3', 'lightpink': '#FFB6C1', 'lightsalmon': '#FFA07A',
              'lightseagreen': '#20B2AA', 'lightskyblue': '#87CEFA', 'lightslategray': '#778899',
              'lightsteelblue': '#B0C4DE', 'lightyellow': '#FFFFE0', 'lime': '#00FF00', 'limegreen': '#32CD32',
              'linen': '#FAF0E6', 'magenta': '#FF00FF', 'maroon': '#800000', 'mediumaquamarine': '#66CDAA',
              'mediumblue': '#0000CD', 'mediumorchid': '#BA55D3', 'mediumpurple': '#9370DB',
              'mediumseagreen': '#3CB371', 'mediumslateblue': '#7B68EE', 'mediumspringgreen': '#00FA9A',
              'mediumturquoise': '#48D1CC', 'mediumvioletred': '#C71585', 'midnightblue': '#191970',
              'mintcream': '#F5FFFA', 'mistyrose': '#FFE4E1', 'moccasin': '#FFE4B5', 'navajowhite': '#FFDEAD',
              'navy': '#000080', 'oldlace': '#FDF5E6', 'olive': '#808000', 'olivedrab': '#6B8E23', 'orange': '#FFA500',
              'orangered': '#FF4500', 'orchid': '#DA70D6', 'palegoldenrod': '#EEE8AA', 'palegreen': '#98FB98',
              'paleturquoise': '#AFEEEE', 'palevioletred': '#DB7093', 'papayawhip': '#FFEFD5', 'peachpuff': '#FFDAB9',
              'peru': '#CD853F', 'pink': '#FFC0CB', 'plum': '#DDA0DD', 'powderblue': '#B0E0E6', 'purple': '#800080',
              'red': '#FF0000', 'rosybrown': '#BC8F8F', 'royalblue': '#4169E1', 'saddlebrown': '#8B4513',
              'salmon': '#FA8072', 'sandybrown': '#FAA460', 'seagreen': '#2E8B57', 'seashell': '#FFF5EE',
              'sienna': '#A0522D', 'silver': '#C0C0C0', 'skyblue': '#87CEEB', 'slateblue': '#6A5ACD',
              'slategray': '#708090', 'snow': '#FFFAFA', 'springgreen': '#00FF7F', 'steelblue': '#4682B4',
              'tan': '#D2B48C', 'teal': '#008080', 'thistle': '#D8BFD8', 'tomato': '#FF6347', 'turquoise': '#40E0D0',
              'violet': '#EE82EE', 'wheat': '#F5DEB3', 'white': '#FFFFFF', 'whitesmoke': '#F5F5F5', 'yellow': '#FFFF00',
              'yellowgreen': '#9ACD32'}
    for i in range(int(class_num)):
        plt.scatter(X_tsne[y == i][:, 0], X_tsne[y == i][:, 1], s=3, alpha=0.5, label=f'{i}',
                    c=list(colors.values())[i])

    # plt.legend(loc='upper left',fontsize=16)
    plt.savefig(save_path + "/" + fileName + '.eps')
    plt.savefig(save_path + "/" + fileName + '.png')
    plt.savefig(save_path + "/" + fileName + '.pdf')
    plt.savefig(save_path + "/" + fileName + '.svg')
    # plt.show()


if __name__ == '__main__':
    # inagenet
    # num = 2000
    # basic_class = 30
    # name = "c" + str(basic_class) + "-p1-b16-n5000-bz250-im-all-v1"
    # # name = "c"+str(basic_class)+"-p1-b16-n5000-bz250-im-DIHN-v1"
    # style = "incre"
    # union_path = os.path.join('../iteratorData', name, style, "final/")
    # savePath = "./"
    # fileName = name
    #
    # seen_b = torch.load(union_path + "seen_b.t")
    # seen_targets = torch.load(union_path + "seen_targets.t")
    # unseen_b = torch.load(union_path + "unseen_b.t")
    # unseen_targets = torch.load(union_path + "unseen_targets.t")
    # all_b = torch.cat((seen_b, unseen_b), dim=0)
    # all_targets = torch.cat((seen_targets, unseen_targets), dim=0)
    # # saveScatterImage(seen_b, seen_targets, num, savePath, fileName+"-basic",basic_class)
    # saveScatterImage(all_b, all_targets, num, savePath, fileName + "-incre", 100)

    #cifar10
    num=2500
    basic_class = 3
    #name = "c3-p1-b32-n2500-bz250-cf-all-v1"
    name = "c3-p1-b16-n2500-bz250-cf-DIHN-v1"
    savePath = "./"
    fileName = name
    union_path1 = os.path.join('../iteratorData',name , "basic")
    seen_b = torch.load(union_path1+"/30-seen_all_u.t").sign()
    seen_targets = torch.load(union_path1+"/30-seen_all_targets.t")

    union_path2 = os.path.join('../iteratorData',name , "incre")
    all_b = torch.load(union_path2+"/90-all_u.t").sign()
    all_targets = torch.load(union_path2+"/90-all_u_targets.t")

    saveScatterImage(seen_b, seen_targets, num, savePath, fileName+"-basic",10)
    saveScatterImage(all_b, all_targets, num, savePath, fileName+"-incre", 10)

