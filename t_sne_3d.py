import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt_sne
from sklearn import datasets
from sklearn.manifold import TSNE
import os
from mpl_toolkits.mplot3d import Axes3D  # 进行3D图像绘制
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
import seaborn as sns



def plot_tsne3d(features, labels, epoch, fileNameDir=None, num_classes=3, labels_list=None):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    # 判断文件夹是否存在，不存在则创建
    if not os.path.exists(fileNameDir):
        os.makedirs(fileNameDir)
    # 指定3维，并初始化
    tsne = TSNE(n_components=3, init='pca', random_state=0)

    try:
        tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    except:
        tsne_features = tsne.fit_transform(features)

    # 对数据进行归一化操作
    x_min, x_max = np.min(tsne_features, 0), np.max(tsne_features, 0)
    embedded = tsne_features / (x_max - x_min)

    # 设置每个散点的大小
    scatter_s = 100  # 设置散点面积

    # 创建颜色调色板
    palette = sns.color_palette(palette="pastel", n_colors=num_classes)
    hex = palette.as_hex()

    # hex = ["#c957db", "#dd5f57", "#b9db57", "#57db30", "#5784db"]  # 粉红，暗红，浅绿，绿，蓝

    # 创建显示的figure
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = [10, 10]
    ax = Axes3D(fig)

    # 将数据对应坐标输入到figure中
    values = zip(embedded[:, 0], embedded[:, 1], embedded[:, 2], labels)
    # v就是标签值：范围[0-4]
    # 根据标签v来为每个类设置不同的颜色
    for x, y, z, v in values:
        if v == 0:
            # x,y,z为坐标值
            # c：用于设定颜色
            # marker：用于指定显示的形状
            # s：设定显示形状的大小
            ax.scatter(x, y, z,
                       # c=np.array(plt.cm.Set1(2)).reshape(1,-1),
                       #  c = "limegreen",
                       c=hex[0],
                       marker=".",
                       s=scatter_s,
                       # label = "c1"
                       )
        if v == 1:
            ax.scatter(x, y, z,
                       # c="lightcoral",
                       c=hex[1],
                       marker=".",
                       s=scatter_s,
                       # label = "c2"
                       )
        if v == 2:
            ax.scatter(x, y, z,
                       # c="coral",
                       c=hex[2],
                       marker=".",
                       s=scatter_s,
                       # label = "c3"
                       )
        if v == 3:
            ax.scatter(x, y, z,
                       # c="slategrey",
                       c=hex[3],
                       marker=".",
                       s=scatter_s,
                       # label = "c4"
                       )
        if v == 4:
            ax.scatter(x, y, z,
                       # c="cadetblue",
                       c=hex[4],
                       marker=".",
                       s=scatter_s,
                       # label = "c5"
                       )

    myHandle = [
        Line2D([], [], marker='.', color=hex[0], markersize=10, linestyle='None'),
        Line2D([], [], marker='.', color=hex[1], markersize=10, linestyle='None'),
        Line2D([], [], marker='.', color=hex[2], markersize=10, linestyle='None'),
        # Line2D([], [], marker='.', color=hex[3], markersize=10, linestyle='None'),
        # Line2D([], [], marker='.', color=hex[4], markersize=10, linestyle='None'),
    ]
    # 用于标签显示的设置
    legend = ax.legend(handles=myHandle,
                       # labels=['real', 'print1', 'print2', 'replay1','replay2'],
                       labels=labels_list,
                       loc="upper right",
                       title="",
                       bbox_to_anchor=(1.0, 1.0),
                       # title_fontsize=20,
                       # prop={'size': 10}
                       )
    ax.add_artist(legend)
    # 保存高清图像到指定的文件夹里面
    plt.savefig(os.path.join(fileNameDir, "%s.jpg") % str(epoch), format="jpg", dpi=300)
    plt.close()



digits = datasets.load_digits(n_class=3)
features, labels = digits.data, digits.target
# print(features.shape)
# print(labels.shape)
file_name = "test"
file_dir = 'test'
labels_list = ['1', '2', '3']


if __name__ == '__main__':
    plot_tsne3d(features=features, labels=labels, epoch=file_name, fileNameDir=file_dir, num_classes=3, labels_list=labels_list)
