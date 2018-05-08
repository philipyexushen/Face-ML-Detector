from lxml import etree
import os
from optparse import OptionParser
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="train_data_result_path", help="Path to training data result.")
    (options, args) = parser.parse_args()
    train_data_result_path = options.train_data_result_path

    files = etree.parse(train_data_result_path)
    data = files.getroot()
    results = data.findall('p')

    lst_mean_overlapping_bboxes = []
    lst_class_acc = []
    lst_loss_rpn_cls = []
    lst_loss_rpn_regr = []
    lst_loss_class_cls = []
    lst_loss_class_regr = []
    lst_total_loss = []

    for result_obj in results:
        try:
            lst_mean_overlapping_bboxes.append(result_obj.find('mean_overlapping_bboxes').text)
            lst_class_acc.append(float(result_obj.find('class_acc').text))
            lst_loss_rpn_cls.append(float(result_obj.find('loss_rpn_cls').text))
            lst_loss_rpn_regr.append(float(result_obj.find('loss_rpn_regr').text))
            lst_loss_class_cls.append(float(result_obj.find('loss_class_cls').text))
            lst_loss_class_regr.append(float(result_obj.find('loss_class_regr').text))
            lst_total_loss.append(float(result_obj.find('total_loss').text))
        except Exception as e:
            print(e)
            continue

    lst_mean_overlapping_bboxes = np.array(lst_mean_overlapping_bboxes)
    lst_class_acc = np.array(lst_class_acc)
    lst_loss_rpn_cls = np.array(lst_loss_rpn_cls)
    lst_loss_rpn_regr = np.array(lst_loss_rpn_regr)
    lst_loss_class_cls = np.array(lst_loss_class_cls)
    lst_loss_class_regr = np.array(lst_loss_class_regr)
    lst_total_loss = np.array(lst_total_loss)

    plt.figure(1)
    # plt.plot(lst_mean_overlapping_bboxes, 'red', label="mean_overlapping_bboxes")
    # plt.plot(lst_class_acc, 'orange', label="class_acc")
    plt.plot(lst_loss_rpn_cls, 'blue', label="loss_rpn_cls")
    plt.plot(lst_loss_rpn_regr, 'brown', label="loss_rpn_regr")
    plt.plot(lst_loss_class_cls, 'red', label="loss_class_cls")
    plt.plot(lst_loss_class_regr, 'purple', label="loss_class_regr")
    plt.plot(lst_total_loss, 'orange', label="total_loss")
    plt.xlabel('epoch'), plt.ylabel('value'), plt.grid(), plt.legend()

    plt.figure(2)
    plt.plot(lst_class_acc, 'orange', label="class_acc")
    plt.xlabel('epoch'), plt.ylabel('class_acc'), plt.grid(), plt.legend()

    plt.show()

