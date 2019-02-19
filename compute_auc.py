from sklearn import metrics
import numpy as np
import pandas as pd

data_dir="data"
output_dir="guba_output"

data = pd.read_csv(data_dir+"/samples_new.csv",dtype=str)
#data = data[:4470]
#data = data[data['type']=='reply']
data = data[7000:]
#print(data)

ytrue = []
for i in data.index:
    d = data.loc[i]
    label1 = d['是否股评相关']
    label2 = d['明天以后看好程度']
    if label1 == '1' and (label2 == '1' or label2 == '2'):
        ytrue.append(1)
    elif label1 == '1' and (label2 == '4' or label2 == '5'):
        ytrue.append(3)
    elif label1 == '1' and label2 == '3':
        ytrue.append(2)
    else:
        ytrue.append(0)

pred_file = open(output_dir+"/test_results.tsv")
ypred = []
for line in pred_file.read().splitlines():
    ldata = line.split('\t')
    ldata = [float(f) for f in ldata]
    ypred.append(ldata.index(max(ldata)))

data['pred'] = ypred
data['label'] = ytrue
print(data[data['pred'] != data['label']])

y_true = ytrue
y_pred = ypred


#####
# Do classification task, 
# then get the ground truth and the predict label named y_true and y_pred
classify_report = metrics.classification_report(y_true, y_pred)
confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
overall_accuracy = metrics.accuracy_score(y_true, y_pred)
acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
average_accuracy = np.mean(acc_for_each_class)
score = metrics.accuracy_score(y_true, y_pred)
print('classify_report : \n', classify_report)
print('confusion_matrix : \n', confusion_matrix)
print('acc_for_each_class : \n', acc_for_each_class)
print('average_accuracy: {0:f}'.format(average_accuracy))
print('overall_accuracy: {0:f}'.format(overall_accuracy))
print('score: {0:f}'.format(score))
