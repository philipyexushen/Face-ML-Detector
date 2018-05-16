import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

if __name__ == "__main__":
    y_true = np.array([1, 1, 0, 0])
    y_scores = np.array([0.9, 0.1, 0.2, 0.3])

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    plt.figure(1)
    plt.subplot(121)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(ap))

    plt.subplot(122)
    plt.step(fpr, tpr, color='r', alpha=0.2, where='post')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('roc curve: auc={0:0.2f}'.format(auc))

    print(precision, recall, thresholds)
    print("ap={0:0.2f}".format(ap))

    print(fpr, tpr, thresholds_roc)
    print("auc={0:0.2f}".format(auc))

    plt.show()