
import itertools
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt
from triton.language import dtype


def plot_confusion_matrix(cm, classes, normalize=False, title='State transition matrix', cmap=plt.cm.Blues):
    plt.figure()
    sns.set(font_scale=0.8)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")

    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")

    plt.ylabel('Self patt')
    plt.xlabel('Transition patt')

    plt.tight_layout()
    plt.savefig('method_2.png', transparent=True, dpi=1000)

    plt.show()


trans_mat = np.array([ (34,   0,   0,   7,   0,   0,   0,  0,   1),
                       (  0,  36,   0,   0,   0,   0,   0,   0,  0),
 (  0,   1, 310,   1,   0,   0,   0,  13,   4),
 (  4,   0,   0,  24,   0,   1,   0,   0,   0),
 ( 0,   0,   1,   0,  54,   0,   0,   0,   0),
 (  0,   0,   0,   0,   0,  35,   0,   0,   0),
 (  1,   0,   0,   1,   0,   1, 119,   0,   9),
 (  1,  0,   1,   3,   0,  0,   0,284,  2),
                       (  0,   0,  2,  0,  6,  0,  1,   6, 537)], dtype=int)


trans_mat1 = np.array([(101,  0,   0,   0,   6),
                        ( 0, 112,   1,   0,   1),
 (  2,   0,  90,   0,   2),
 (  0,   1,   0,  86,   0),
                        (  4,   0,   2,   0, 92)], dtype=int)


trans_mat1 = np.array([(101,  0,   0,   0,   6),
                        ( 0, 112,   1,   0,   1),
 (  2,   0,  90,   0,   2),
 (  0,   1,   0,  86,   0),
                        (  4,   0,   2,   0, 92)], dtype=int)

trans_Resnet50 = np.array([(1.770e+02, 5.000e+00, 5.000e+00, 2.100e+01, 3.000e+00, 4.000e+00, 0.000e+00,
  1.100e+01, 4.000e+00),
 (1.000e+00 ,1.630e+02, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00 ,0.000e+00,
  0.000e+00, 0.000e+00),
(2.000e+00 ,0.000e+00 ,1.492e+03 ,2.000e+00, 0.000e+00 ,0.000e+00 ,1.800e+01,
  4.800e+01 ,1.110e+02),
 (3.700e+01, 1.000e+00, 0.000e+00 ,1.420e+02 ,2.000e+00, 3.000e+00, 4.000e+00,
  0.000e+00, 2.000e+00),
 (2.000e+00 ,0.000e+00, 0.000e+00 ,1.000e+00 ,1.460e+02 ,5.000e+00 ,1.000e+00,
  7.000e+00 ,7.100e+01),
 (1.000e+01 ,3.000e+00 ,5.000e+00, 8.000e+00 ,1.000e+00 ,1.790e+02 ,0.000e+00,
  4.000e+00, 5.000e+00),
 (0.000e+00 ,0.000e+00, 3.400e+01 ,2.000e+00 ,0.000e+00 ,1.000e+00 ,4.660e+02,
  1.000e+00 ,5.400e+01),
 (1.500e+01 ,4.000e+00 ,3.400e+01 ,4.000e+00 ,1.400e+01 ,2.000e+00 ,6.000e+00,
  1.296e+03 ,3.300e+01),
 (2.000e+00 ,0.000e+00 ,8.500e+01 ,2.000e+00 ,8.700e+01 ,8.000e+00 ,8.400e+01,
  4.800e+01 ,2.512e+03)],dtype=int)

trans_zh = np.array([(7.800e+01, 6.000e+00 ,1.000e+00 ,2.800e+01 ,7.000e+00 ,1.000e+01, 0.000e+00,
  6.000e+00 ,2.000e+00),
 (5.000e+00 ,9.400e+01 ,0.000e+00, 8.000e+00, 0.000e+00 ,7.000e+00 ,0.000e+00,
  0.000e+00, 0.000e+00),
 (0.000e+00, 0.000e+00 ,8.610e+02, 0.000e+00, 0.000e+00 ,0.000e+00 ,2.900e+01,
  2.500e+01 ,3.700e+01),
 (2.500e+01 ,3.000e+00 ,0.000e+00 ,7.600e+01 ,0.000e+00 ,1.100e+01 ,0.000e+00,
  1.000e+00 ,0.000e+00),
(1.000e+00 ,0.000e+00, 0.000e+00 ,3.000e+00, 1.000e+02 ,5.000e+00 ,0.000e+00,
  8.000e+00, 1.700e+01),
(4.000e+00, 7.000e+00, 0.000e+00, 9.000e+00, 0.000e+00 ,6.400e+01 ,0.000e+00,
  1.000e+00, 0.000e+00),
 (1.000e+00 ,0.000e+00, 1.100e+01, 0.000e+00 ,1.000e+00, 0.000e+00 ,2.720e+02,
  9.000e+00 ,1.800e+01),
 (1.500e+01 ,3.000e+00 ,1.800e+01 ,6.000e+00 ,2.500e+01 ,7.000e+00 ,1.400e+01,
  7.350e+02 ,2.100e+01),
 (6.000e+00, 1.000e+00 ,1.040e+02, 1.000e+00, 3.300e+01, 5.000e+00, 3.200e+01,
  3.800e+01 ,1.585e+03)], dtype=int)

trans_zhs = np.array([( 94,  0,   1,   0,   5),
 (  0,  88,   0,   0,   0),
 (  1,   0,  99,   1,   2),
 (  1,   0,   2, 100,   0),
 ( 2,   0,   3,   0, 101)], dtype=int)
"""method 2"""
if True:
    labels1= ['1','2','3','4','5']
    labels2 = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    label = labels1
    plot_confusion_matrix(trans_zhs, label)
