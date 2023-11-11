import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
import numpy as np
# from deepfefm import DeepFEFM
# from autoint import  AutoInt
# from xdeepfm import xDeepFM
from self_AtDFEFM import self_AtDFEFM
from feature_column import SparseFeat, DenseFeat, get_feature_names

if __name__ == "__main__":
    data = pd.read_csv('/home/user/PycharmProjects/yyy/deepfefm/criteo_sample.txt')

    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]

    data[sparse_features] = data[sparse_features].fillna('-1', )  # 类别特征缺失 ，使用-1代替
    data[dense_features] = data[dense_features].fillna(0, )  # 数值特征缺失，使用0代替
    target = ['label']  # label

    # 归一化
    data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])


    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=10)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.1, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    # This callback will stop the training when there is no improvement in
    # 4.Define Model,train,predict and evaluate
    model = self_AtDFEFM(linear_feature_columns, dnn_feature_columns, task='binary')
    # model.summary()

    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=1024, epochs=100, verbose=2, callbacks=[callback], validation_split=0.1, )
    pred_ans = model.predict(test_model_input, batch_size=1024)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    print("test Accuracy", round(accuracy_score(test[target].values, pred_ans), 4))
    # print("test F1", round(f1_score(test[target].values, pred_ans), 4))


