from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import math
from xlsx_file_writer import CXlsFileWriter

class PrivateMethod:
    @staticmethod
    def union(au, bu, area_intersection):
        area_a = (au[2] - au[0]) * (au[3] - au[1])
        area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
        area_union = area_a + area_b - area_intersection
        return area_union

    @staticmethod
    def intersection(ai, bi):
        x = max(ai[0], bi[0])
        y = max(ai[1], bi[1])
        w = min(ai[2], bi[2]) - x
        h = min(ai[3], bi[3]) - y
        if w < 0 or h < 0:
            return 0
        return w*h

    @staticmethod
    def iou(a, b):
        # a and b should be (x1,y1,x2,y2)

        if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
            return 0.0

        area_i = PrivateMethod.intersection(a, b)
        area_u = PrivateMethod.union(a, b, area_i)

        # IOU的求法，标准的交集/并集，分母加上一个很小的数防止除以0
        return float(area_i) / float(area_u + 1e-6)

    @staticmethod
    def make_binary_list(T_real, P_real, key, original_mapping):
        T_bin_real = []
        P_bin_real = []

        for T_item, P_item in zip(T_real, P_real):
            T_is_key: bool = False
            P_is_key: bool = False

            if T_item == original_mapping[key]:
                T_is_key = True
            if P_item == original_mapping[key]:
                P_is_key = True

            T_bin_real.append(int(T_is_key))
            P_bin_real.append(int(P_is_key))
        return T_bin_real, P_bin_real


def get_map(pred, gt, f, original_mapping):
    T = {}
    P = {}

    T_real = []
    P_real = []

    fx, fy = f

    for bbox in gt:
        bbox['bbox_matched'] = False

    pred_probs = np.array([s['prob'] for s in pred])
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_y1 = pred_box['y1']
        pred_y2 = pred_box['y2']
        pred_prob = pred_box['prob']
        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        found_match = False

        max_iou = 0
        most_pred_class = 'bg'
        for gt_box in gt:
            gt_class = gt_box['class']
            gt_x1 = gt_box['x1']/fx
            gt_x2 = gt_box['x2']/fx
            gt_y1 = gt_box['y1']/fy
            gt_y2 = gt_box['y2']/fy
            gt_seen = gt_box['bbox_matched']

            iou = PrivateMethod.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            if iou > max_iou:
                max_iou = iou
                most_pred_class = gt_class

            if gt_class != pred_class:
                continue
            if gt_seen:
                continue

            if iou >= 0.5:
                found_match = True
                gt_box['bbox_matched'] = True
                break
            else:
                continue

        T[pred_class].append(int(found_match))

        T_real.append(int(original_mapping[most_pred_class]))
        P_real.append(int(original_mapping[pred_class]))

    for gt_box in gt:
        if not gt_box['bbox_matched'] and not gt_box['difficult']:
            if gt_box['class'] not in P:
                pass
                # P[gt_box['class']] = []
                # T[gt_box['class']] = []

            # T[gt_box['class']].append(1)
            # P[gt_box['class']].append(0)

            T_real.append(int(original_mapping[gt_box['class']]))
            P_real.append(int(original_mapping['bg']))

    return T, P, T_real, P_real

def output_ap(T, P):
    all_aps = []
    for key in T.keys():
        ap = average_precision_score(T[key], P[key])
        if math.isnan(ap):
            continue

        print('{} AP: {}'.format(key, ap))
        all_aps.append(ap)
    print('mAP = {}'.format(np.mean(np.array(all_aps))))

def draw_measure_curve(T:dict, P:dict, T_real, P_real, original_mapping):
    figure_count = 0

    for key in T.keys():
        T_bin_real, P_bin_real = PrivateMethod.make_binary_list(T_real, P_real, key, original_mapping)

        f1 = f1_score(T_bin_real, P_bin_real)
        recall = recall_score(T_bin_real, P_bin_real)
        precision = precision_score(T_bin_real, P_bin_real)
        print(f"key={key}, recall={recall}, precision={precision}, f1={f1}")

        plt.figure(figure_count)
        plt.subplot(121)
        ap = average_precision_score(T[key], P[key])
        precision, recall, _ = precision_recall_curve(T[key], P[key])
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: AP={0:0.2f}'.format(ap))

        plt.subplot(122)
        auc = roc_auc_score(T[key], P[key])
        fpr, tpr, _ = roc_curve(T[key], P[key])
        plt.step(fpr, tpr, color='r', alpha=0.2, where='post')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('roc curve: auc={0:0.2f}'.format(auc))

        figure_count += 1

    # 多分类
    f1 = f1_score(T_real, P_real, average="macro")
    recall = recall_score(T_real, P_real, average="macro")
    precision = precision_score(T_real, P_real, average="macro")
    print(f"key=multi, recall={recall}, precision={precision}, f1={f1}")

    plt.show()