'''
Confusion matrix（with Times New Roman）
'''
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# plt.rc('font', family='Times New Roman')
end_all_targets = np.load('/****/all_targets.npy')
end_all_outputs = np.load('/****/all_outputs.npy')

sns.set()
f, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
plt.rcParams['font.sans-serif'] = ['Times New Roman']
mcm = multilabel_confusion_matrix(np.array(end_all_targets), np.array(end_all_outputs))
sign_name_box = np.array([['TLI', 'Lobulated', 'Spiculated'], ['VC', 'AB', 'PI']])
for index, (name, sub_ax) in enumerate(zip(sign_name_box.flatten(), ax.flatten())):
    sns.heatmap(mcm[index], annot=True, fmt='.20g', ax=sub_ax, cmap='Blues', cbar=False, annot_kws={"fontsize": 20})
    sub_ax.set_title(name, fontsize=22)  # 标题
    sub_ax.set_xlabel('Predicted label', fontsize=16)  # x轴
    sub_ax.set_ylabel('True label', fontsize=16)  # y轴
plt.tight_layout()
plt.show()
print()