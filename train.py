import numpy as np
# sys.path.append('../input')

import os

# mfe256 256 128 111 e30 fn0.001

from Utils import *
from DataGenerator import *


Path = "./datasets/ISRUC/ISRUC_S3/ISRUC_S3_all.npz"
output_path = "./MFE/output_s3/"


ReadList = np.load(Path, allow_pickle=True)
Fold_Num   = ReadList['Fold_len']    # Num of samples of each fold
Fold_Data  = ReadList['Fold_data']   # Data of each fold
Fold_Label = ReadList['Fold_label']  # Labels of each fold


freq = 100
channels = 10
subject_num = len(Fold_Num)
fold = 10


cfg = {
    'bs': 32,
    'epochs': 50
}

DataGenerator = kFoldGenerator(Fold_Data, Fold_Label, fold, subject_num)
del ReadList, Fold_Label

print('y_list length:', len(DataGenerator.y_list))

if not os.path.exists(output_path):
    os.makedirs(output_path)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Activation,\
BatchNormalization, Add, Reshape, TimeDistributed, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

from sklearn import metrics

import random as python_random
# os.environ['TF_DETERMINISTIC_OPS'] = '0'
np.random.seed(32)
python_random.seed(32)
tf.random.set_seed(32)
# print("keras version:", keras.__version__)
 
# print(device_lib.list_local_devices())

tf.config.set_soft_device_placement(True)

# 获取所有可用的 GPU 设备
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Exclude GPU:2 and set GPU:1 as the visible device
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU:0 for computations")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs available. Using CPU for computations.")

# 验证当前使用的设备
print("Current device:", tf.config.get_visible_devices('GPU'))



from tensorflow.keras.layers import SeparableConv1D

def CNN_light(inputs, fs=8, kernel_size=3, pool_size=2, weight=0.001):
    x = SeparableConv1D(fs, kernel_size, 1, padding='same', kernel_regularizer=l2(weight))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size, 2, padding='same')(x)
    return x

def ResNet_light(inputs, fs=8, ks_1=3, ps_1=2, ks_2=3, ps_2=2, weight=0.001):
    x = CNN_light(inputs, fs, ks_1, ps_1, weight)
    x = CNN_light(x, fs, ks_2, ps_2, weight)
    shortcut_x = SeparableConv1D(fs, 1, 2, padding='same')(inputs)
    shortcut_x = SeparableConv1D(fs, 1, 2, padding='same')(shortcut_x)
    return Add()([x, shortcut_x])

def create_model_light(input_shape, channels=10, time_second=30, freq=100):
    inputs_channel = Input(shape=(time_second*freq, 1))
    x = ResNet_light(inputs_channel, 8)
    x = Dropout(0.1)(x)
    x = ResNet_light(x, 16)
    x = Dropout(0.1)(x)
    x = ResNet_light(x, 32)
    x = Dropout(0.1)(x)

    outputs = GlobalAveragePooling1D()(x)

    fea_part = Model(inputs=inputs_channel, outputs=outputs)
    inputs = Input(shape=input_shape)  # (3000, 10)
    input_re = Reshape((channels, time_second*freq, 1))(inputs)  # (10, 3000, 1)
    fea_all = TimeDistributed(fea_part)(input_re)

    fla_fea = Flatten()(fea_all)
    fla_fea = Dropout(0.3)(fla_fea)

    merged = Dense(64)(fla_fea)
    label_out = Dense(5, activation='softmax', name='Label')(merged)

    ce_model = Model(inputs, label_out)
    ce_model.summary()
    return ce_model




import gc
# k-fold cross validation
all_scores = []


first_decay_steps = 10
lr_decayed_fn = (
  tf.keras.optimizers.schedules.CosineDecayRestarts(
      0.001,
      first_decay_steps))


tf.config.experimental_run_functions_eagerly(True)


best_val_acc = []
all_scores = []

for i in range(0, fold):  # 20-fold
    print('Fold #', i)

    train_data, train_targets, val_data, val_targets = DataGenerator.getFold(i)  # train_data [7665, 10, 3000] val_data[924, 10, 3000] train_targets[7665, 5] val_targets[924, 5]
    train_data, val_data = train_data.reshape(-1, 30 * freq, channels), val_data.reshape(-1, 30 * freq, channels)  # train_data [7665, 3000, 10] val_data[924, 3000, 10]
    

    opt = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn, amsgrad=True)
    ce_model = create_model_light(input_shape=train_data.shape[1:], freq=freq, channels=channels, time_second=30)
    verbose = 1

    ce_model.compile(
        optimizer=opt,
        loss={'Label': "categorical_crossentropy"},
        metrics={'Label': "accuracy"}
    )
    # ce_model.summary()
    if not os.path.exists(output_path+str(i)+'ResNet_Best'+'.h5'):

        history = ce_model.fit(
            train_data, train_targets,
            batch_size=cfg['bs'], epochs=cfg['epochs'],
            validation_data=(val_data, val_targets),
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    output_path + str(i) + 'ResNet_Best' + '.h5',
                    monitor='val_accuracy',  # 监控验证准确率
                    verbose=1,
                    save_best_only=True,
                    mode='auto')],
            verbose=verbose
        )

        # Save training information
        fit_loss = np.array(history.history['loss'])
        fit_acc = np.array(history.history['accuracy'])
        fit_val_loss = np.array(history.history['val_loss'])
        fit_val_acc = np.array(history.history['val_accuracy'])
        print('Best val acc:', max(history.history['val_accuracy']))
        best_val_acc.append(max(history.history['val_accuracy']))

        saveFile = open(output_path + "Result_MFE.txt", 'a+')
        print('Fold #'+str(i), file=saveFile)
        print(history.history, file=saveFile)
        saveFile.close()    


    # get and save the learned feature
    ce_model.load_weights(output_path+str(i)+'ResNet_Best'+'.h5', skip_mismatch=True, by_name=True)


    # Predict ------------------------------------------------------------
    predictions = ce_model.predict(val_data, batch_size=cfg['bs'])

    AllPred_temp = np.argmax(predictions, axis=1)
    AllTrue_temp = np.argmax(val_targets, axis=1)

    acc = metrics.accuracy_score(AllTrue_temp, AllPred_temp)
    print("Predict accuracy:", acc)
    all_scores.append(acc)

    if i == 0:
        AllPred = AllPred_temp
        AllTrue = AllTrue_temp
    else:
        AllPred = np.concatenate((AllPred, AllPred_temp))
        AllTrue = np.concatenate((AllTrue, AllTrue_temp))

    # VariationCurve(fit_acc, fit_val_acc, f'Acc_{i}', output_path, figsize=(9, 6))
    # VariationCurve(fit_loss, fit_val_loss, f'Loss_{i}', output_path, figsize=(9, 6))

    # Print score to console
    print(128*'=')
    PrintScore(AllTrue_temp, AllPred_temp, fold=i, savePath=output_path)
    ConfusionMatrix(AllTrue_temp, AllPred_temp, fold=i, classes=['W', 'N1', 'N2', 'N3', 'REM'], savePath=output_path)

    # Fold finish
    keras.backend.clear_session()
    del train_data, train_targets, val_data, val_targets
    gc.collect()
    print('Fold #', i, 'finished')



print(128 * '_')
print('End of training MFE.')
print(128 * '#')


print(128*'=')
print("All folds' acc: ",all_scores)
print("Average acc of each fold: ",np.mean(all_scores))

# Print score to console
print(128*'=')
PrintScore(AllTrue, AllPred, savePath=output_path)

# Print confusion matrix and save
ConfusionMatrix(AllTrue, AllPred, classes=['W','N1','N2','N3','REM'], savePath=output_path)

print('End of evaluating MFE.')
print('###train without contrastive learning###')
print(128 * '#')

# 清理数据
del Fold_Data
gc.collect()



