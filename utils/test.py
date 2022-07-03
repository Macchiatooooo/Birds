import numpy as np
import tensorflow as tf
from sklearn import metrics
# from tensorflow.keras import metrics
from tensorflow.python import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def data_load(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        # 数据集划分比例
        validation_split=0.2,
        # 选择训练集
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        # 数据集划分比例
        validation_split=0.2,
        # 选择验证集
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = val_ds.class_names

    return train_ds, val_ds, class_names
train_ds, val_ds, class_names = data_load("C:/Users/macchiato/Desktop/keshe/birds_tf2.3-master/data", 224, 224, 4)
model = keras.models.load_model("C:/Users/macchiato/Desktop/keshe/birds_tf2.3-master/models/mobilenet_birds.h5", compile=True)

y_pred = np.array([])
y_test = np.array([])
x_test = np.array([])
for x,y in val_ds:
    y_pred = np.concatenate([y_pred, np.argmax(model.predict(x), axis=-1)])
    y_test = np.concatenate([y_test, np.argmax(y.numpy(), axis=-1)])

report = metrics.classification_report(y_test, y_pred)  # 获得分类报告
print('输出分类报告：\n', report)
# 混淆矩阵如下
# cm = confusion_matrix(y_pred, y_test)
# print('混淆矩阵为:')
# print(cm)
print('精确度Precision:', metrics.precision_score(y_test, y_pred, average='macro'))
print('召回率Recall:', metrics.recall_score(y_test, y_pred, average='macro'))

# 混淆矩阵可视化
def plotcm(cm):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    classes = [str(i) for i in range(10)]
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y),
                         horizontalalignment='center',
                         verticalalignment='center')
    plt.grid(True, which='minor', linestyle='-')

cm = confusion_matrix(y_test, y_pred)  # 混淆矩阵
print("混淆矩阵:\n", cm)
plotcm(cm)  # 绘制混淆矩阵
plt.show()