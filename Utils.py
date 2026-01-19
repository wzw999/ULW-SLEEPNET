import configparser
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.sparse.linalg import eigs
from tensorflow import keras

# 配置matplotlib以优化内存管理
plt.rcParams['figure.max_open_warning'] = 30  # 增加警告阈值
plt.ioff()  # 关闭交互模式，避免在后台保留图形



##########################################################################################
# Print score between Ytrue and Ypred ####################################################

def PrintScore(true, pred, fold=-1, savePath=None, average='macro', model_name="model"):
    # savePath=None -> console, else to Result.txt
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + f"Result_{model_name}.txt", 'a+')
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores for fold ", str(fold))
    print('Acc\tF1S\tKappa\tF1_W\tF1_N1\tF1_N2\tF1_N3\tF1_R', file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' %
          (metrics.accuracy_score(true, pred),
           metrics.f1_score(true, pred, average=average),
           metrics.cohen_kappa_score(true, pred),
           F1[0], F1[1], F1[2], F1[3], F1[4]),
          file=saveFile)
    # Classification report
    print("\nClassification report:", file=saveFile)
    print(metrics.classification_report(true, pred,
                                        target_names=['Wake','N1','N2','N3','REM'],
                                        digits=4), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true,pred), file=saveFile)
    # Overall scores
    print('\n    Accuracy\t',metrics.accuracy_score(true,pred), file=saveFile)
    print(' Cohen Kappa\t',metrics.cohen_kappa_score(true,pred), file=saveFile)
    print('    F1-Score\t',metrics.f1_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    print('   Precision\t',metrics.precision_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    print('      Recall\t',metrics.recall_score(true,pred,average=average), '\tAverage =',average, file=saveFile)
    if savePath != None:
        saveFile.close()
    return

##########################################################################################
# Print confusion matrix and save ########################################################

def ConfusionMatrix(y_true, y_pred, classes, savePath, fold=-1, model_name="model", title=None, cmap=plt.cm.Blues):
    if not title:
        title = f'Confusion matrix for fold {str(fold)}'
    
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_n=cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(5, 4))
    
    try:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j]*100,'.2f')+'%\n'+format(cm_n[i, j],'d'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(savePath + f"{model_name}_ConfusionMatrix_fold_{fold}.png", 
                   bbox_inches='tight', dpi=150)  # 添加bbox_inches和dpi参数优化保存
        # plt.show()  # 注释掉以避免阻塞训练进程
    finally:
        # 确保图形被关闭，即使出现异常
        plt.close(fig)
    
    return ax

##########################################################################################
# Draw ACC / loss curve and save #########################################################

def VariationCurve(fit,val,yLabel,savePath,figsize=(9, 6)):
    # 显式创建figure对象
    fig = plt.figure(figsize=figsize)
    
    try:
        plt.plot(range(1,len(fit)+1), fit,label='Train')
        plt.plot(range(1,len(val)+1), val, label='Val')
        plt.title('Model ' + yLabel)
        plt.xlabel('Epochs')
        plt.ylabel(yLabel)
        plt.legend()
        plt.tight_layout()  # 添加tight_layout优化布局
        plt.savefig(savePath + 'Model_' + yLabel + '.png', 
                   bbox_inches='tight', dpi=150)  # 添加bbox_inches和dpi参数
        # plt.show()  # 注释掉以避免阻塞训练进程
    finally:
        # 确保图形被关闭，即使出现异常
        plt.close(fig)
    
    return

