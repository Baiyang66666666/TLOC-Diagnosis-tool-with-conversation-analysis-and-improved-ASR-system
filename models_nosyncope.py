import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


parser = argparse.ArgumentParser ()
parser.add_argument ("-m", "--model", help="the model for classify, svm|rf|xb", type=str, default="xb")
args = parser.parse_args ()
model = args.model


def divide_dataset(X, Y):
    shuffle_list = [i for i in range (0, len (X))]
    np.random.shuffle(shuffle_list)
    X_train = [X[shuffle_list[i]] for i in range (int (len (shuffle_list) * 0.8))]
    Y_train = [Y[shuffle_list[i]] for i in range (int (len (shuffle_list) * 0.8))]

    X_dev = [X[shuffle_list[len (X_train) + j]] for j in range (int (len (shuffle_list) * 0.1))]
    Y_dev = [Y[shuffle_list[len (X_train) + j]] for j in range (int (len (shuffle_list) * 0.1))]

    # print(len(X_train)+len(X_dev))
    # print(len(shuffle_list))
    # print(len(Y))
    # print(len(X))
    X_test = [X[shuffle_list[k]] for k in range (len (X_train) + len (X_dev), len (shuffle_list))]
    Y_test = [Y[shuffle_list[k]] for k in range (len (X_train) + len (X_dev), len (shuffle_list))]

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test


def SVM_MODEL(X_train, Y_train, X_dev, Y_dev, X_test, Y_test):
    clf = svm.SVC (C=5, gamma=0.05, max_iter=-1)
    clf.fit (np.array (X_train), np.array (Y_train))
    Y_pred_dev = clf.predict (np.array (X_dev))
    pred_dev = accuracy_score (Y_dev, Y_pred_dev)

    Y_pred_test = clf.predict (np.array (X_test))
    pred_test = accuracy_score (Y_test, Y_pred_test)

    # 获取概率预测值
    Y_dev_probs = clf.predict_proba (np.array (X_dev))
    Y_test_probs = clf.predict_proba (np.array (X_test))

    # 调整阈值
    #Y_dev_pred = adjust_thresholds (Y_dev_probs, thresholds)
    #Y_test_pred = adjust_thresholds (Y_test_probs, thresholds)
    # 将预测数组转换为预测的类别标签
    Y_dev_pred = [np.argmax (pred) for pred in Y_dev_probs]
    Y_test_pred = [np.argmax (pred) for pred in Y_test_probs]

    # 计算精确度（针对每个类别）
    precision_dev = []
    precision_test = []

    for class_idx in range (len (SENTIMENT_NAME_DIC)):
        class_mask_dev = np.array (Y_dev) == class_idx
        class_mask_test = np.array (Y_test) == class_idx

        class_pred_dev = np.array (Y_dev_pred) == class_idx
        class_pred_test = np.array (Y_test_pred) == class_idx

        precision_dev_class = sum (class_pred_dev[class_mask_dev]) / sum (class_mask_dev)
        precision_test_class = sum (class_pred_test[class_mask_test]) / sum (class_mask_test)

        precision_dev.append (precision_dev_class)
        precision_test.append (precision_test_class)

    print ('dev precision per class:', precision_dev)
    print ('test precision per class:', precision_test)

    return precision_dev, precision_test, Y_dev_pred, Y_test_pred, pred_dev, pred_test


def KNN_MODEL(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, k_neighbors=3):
    clf = KNeighborsClassifier (n_neighbors=k_neighbors)
    clf.fit (np.array (X_train), np.array (Y_train))
    Y_pred_dev = clf.predict (np.array (X_dev))
    pred_dev = accuracy_score (Y_dev, Y_pred_dev)

    Y_pred_test = clf.predict (np.array (X_test))
    pred_test = accuracy_score (Y_test, Y_pred_test)

    # 获取概率预测值
    Y_dev_probs = clf.predict_proba (np.array (X_dev))
    Y_test_probs = clf.predict_proba (np.array (X_test))

    # 调整阈值
    #Y_dev_pred = adjust_thresholds (Y_dev_probs, thresholds)
    #Y_test_pred = adjust_thresholds (Y_test_probs, thresholds)
    # 将预测数组转换为预测的类别标签
    Y_dev_pred = [np.argmax (pred) for pred in Y_dev_probs]
    Y_test_pred = [np.argmax (pred) for pred in Y_test_probs]

    # 计算精确度（针对每个类别）
    precision_dev = []
    precision_test = []

    for class_idx in range (len (SENTIMENT_NAME_DIC)):
        class_mask_dev = np.array (Y_dev) == class_idx
        class_mask_test = np.array (Y_test) == class_idx

        class_pred_dev = np.array (Y_dev_pred) == class_idx
        class_pred_test = np.array (Y_test_pred) == class_idx

        precision_dev_class = sum (class_pred_dev[class_mask_dev]) / sum (class_mask_dev)
        precision_test_class = sum (class_pred_test[class_mask_test]) / sum (class_mask_test)

        precision_dev.append (precision_dev_class)
        precision_test.append (precision_test_class)

    print ('dev precision per class:', precision_dev)
    print ('test precision per class:', precision_test)

    return precision_dev, precision_test, Y_dev_pred, Y_test_pred, pred_dev, pred_test


def RF_MODEL(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, thresholds=[0.4,0.7]):
    clf = RandomForestClassifier (n_estimators=500, max_depth=32, random_state=8)
    clf.fit (np.array (X_train), np.array (Y_train))
    Y_pred_dev = clf.predict (np.array (X_dev))
    pred_dev = accuracy_score (Y_dev, Y_pred_dev)

    Y_pred_test = clf.predict (np.array (X_test))
    pred_test = accuracy_score (Y_test, Y_pred_test)

    # 获取概率预测值
    Y_dev_probs = clf.predict_proba (np.array (X_dev))
    Y_test_probs = clf.predict_proba (np.array (X_test))

    # 调整阈值
    Y_dev_pred = adjust_thresholds (Y_dev_probs, thresholds)
    Y_test_pred = adjust_thresholds (Y_test_probs, thresholds)
    # 将预测数组转换为预测的类别标签
    Y_dev_pred = [np.argmax (pred) for pred in Y_dev_pred]
    Y_test_pred = [np.argmax (pred) for pred in Y_test_pred]

    # 计算精确度（针对每个类别）
    precision_dev = []
    precision_test = []

    for class_idx in range (len (SENTIMENT_NAME_DIC)):
        class_mask_dev = np.array (Y_dev) == class_idx
        class_mask_test = np.array (Y_test) == class_idx

        class_pred_dev = np.array (Y_dev_pred) == class_idx
        class_pred_test = np.array (Y_test_pred) == class_idx

        precision_dev_class = sum (class_pred_dev[class_mask_dev]) / sum (class_mask_dev)
        precision_test_class = sum (class_pred_test[class_mask_test]) / sum (class_mask_test)

        precision_dev.append (precision_dev_class)
        precision_test.append (precision_test_class)

    print ('dev precision per class:', precision_dev)
    print ('test precision per class:', precision_test)

    return precision_dev, precision_test, Y_dev_pred, Y_test_pred, pred_dev, pred_test

def adjust_thresholds(y_probs, thresholds):
    y_preds = []
    for probs in y_probs:
        y_pred = []
        for i in range(len(probs)):
            if probs[i] >= thresholds[i]:
                y_pred.append(1)
            else:
                y_pred.append(0)
        y_preds.append(y_pred)
    return y_preds

def XGBOOST_MODEL(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, thresholds=[0.4,0.7]):
    class_weight =[1.0,1.0]
    clf = xgb.XGBClassifier (objective="binary:logistic", random_state=42, learning_rate=0.05, max_depth=8,
                             n_estimators=100, subsample=0.5, colsample_bytree=0.5, class_weight=class_weight)
    clf.fit (np.array (X_train), np.array (Y_train))
    Y_pred_dev = clf.predict (np.array (X_dev))
    pred_dev = accuracy_score(Y_dev, Y_pred_dev)

    Y_pred_test = clf.predict (np.array (X_test))
    pred_test = accuracy_score(Y_test, Y_pred_test)

    # 获取概率预测值
    Y_dev_probs = clf.predict_proba (np.array (X_dev))
    Y_test_probs = clf.predict_proba (np.array (X_test))

    # 调整阈值
    Y_dev_pred = adjust_thresholds (Y_dev_probs, thresholds)
    Y_test_pred = adjust_thresholds (Y_test_probs, thresholds)
    # 将预测数组转换为预测的类别标签
    Y_dev_pred = [np.argmax (pred) for pred in Y_dev_pred]
    Y_test_pred = [np.argmax (pred) for pred in Y_test_pred]

    # 计算精确度（针对每个类别）
    precision_dev = []
    precision_test = []

    for class_idx in range (len (SENTIMENT_NAME_DIC)):
        class_mask_dev = np.array (Y_dev) == class_idx
        class_mask_test = np.array (Y_test) == class_idx

        class_pred_dev = np.array (Y_dev_pred) == class_idx
        class_pred_test = np.array (Y_test_pred) == class_idx


        precision_dev_class = sum (class_pred_dev[class_mask_dev]) / sum (class_mask_dev)
        precision_test_class = sum (class_pred_test[class_mask_test]) / sum (class_mask_test)

        precision_dev.append (precision_dev_class)
        precision_test.append (precision_test_class)

    print ('dev precision per class:', precision_dev)
    print ('test precision per class:', precision_test)

    return precision_dev, precision_test, Y_dev_pred, Y_test_pred, pred_dev, pred_test


if __name__ == '__main__':
    SENTIMENT_NAME_DIC = {'Epilepsy': 0, 'FDS': 1}  # 只保留'Epilepsy'和'FDS'
    dataset = pd.read_csv("./misc/label_nosyncope.csv")
    types = dataset.iloc[:, 1].tolist()

    Y = [SENTIMENT_NAME_DIC[type] for type in types]


    X = pd.read_csv("./misc/whisper_LIWC_nosyncope.csv")
    X = X.iloc[:, 2:].values

    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = divide_dataset(X, Y)
    X = np.concatenate ((X_train, X_dev), axis=0)
    Y = np.concatenate ((Y_train, Y_dev), axis=0)

    #k-fold cross validation
    k_folds = 5  # Setting the number of folds for K-fold cross validation

    kf = KFold (n_splits=k_folds, shuffle=True, random_state=42)


    dev_accuracies = []  # Developed set accuracy for storing each fold
    test_accuracies = []
    dev_f1_scores = []
    test_f1_scores = []
    dev_precisions = []
    test_precisions = []
    dev_recalls = []
    test_recalls = []

    # Initialisation dictionaries are used to count the proportion of predictions for each label
    label_predictions_dev = {label_name: [0, 0] for label_name in SENTIMENT_NAME_DIC.keys ()}
    label_predictions_test = {label_name: [0, 0] for label_name in SENTIMENT_NAME_DIC.keys ()}

    for fold, (train_idx, dev_idx) in enumerate (kf.split (X), 1):
        X_train_fold, X_dev_fold = X[train_idx], X[dev_idx]
        Y_train_fold, Y_dev_fold = np.array (Y)[train_idx], np.array (Y)[dev_idx]
        print('dev folder', X_dev_fold)
        if model == "svm":
            print (f'Fold {fold}: start training SVM...')
            dev_accuracy, test_accuracy = SVM_MODEL (X_train_fold, Y_train_fold, X_dev_fold, Y_dev_fold, X_test, Y_test)
        elif model == "rf":
            print (f'Fold {fold}: start training Random Forest...')
            dev_accuracy, test_accuracy = RF_MODEL (X_train_fold, Y_train_fold, X_dev_fold, Y_dev_fold, X_test, Y_test)
        elif model == "xb":
            print (f'Fold {fold}: start training XGBoost...')
            dev_accuracy, test_accuracy, Y_dev_pred, Y_test_pred, dev_accuracy_single,test_accuracy_single = XGBOOST_MODEL (X_train_fold, Y_train_fold, X_dev_fold, Y_dev_fold, X_test,
                                                         Y_test)
        else:
            print (f'Fold {fold}: default using XGBoost...')
            dev_accuracy, test_accuracy, Y_dev_pred, Y_test_pred, dev_accuracy_single,test_accuracy_single = XGBOOST_MODEL(X_train_fold, Y_train_fold, X_dev_fold, Y_dev_fold, X_test,
                                                         Y_test)
        #print('single', dev_accuracy_single)
        dev_accuracies.append (dev_accuracy_single)
        test_accuracies.append (test_accuracy_single)
        dev_precision = precision_score (Y_dev_fold, Y_dev_pred, average='weighted')
        test_precision = precision_score (Y_test, Y_test_pred, average='weighted')
        dev_recall = recall_score (Y_dev_fold, Y_dev_pred, average='weighted')
        test_recall = recall_score (Y_test, Y_test_pred, average='weighted')
        dev_f1 = f1_score (Y_dev_fold, Y_dev_pred, average='weighted')
        test_f1 = f1_score (Y_test, Y_test_pred, average='weighted')

        dev_f1_scores.append (dev_f1)
        test_f1_scores.append (test_f1)
        dev_precisions.append (dev_precision)
        test_precisions.append (test_precision)
        dev_recalls.append (dev_recall)
        test_recalls.append (test_recall)

        # Calculate the proportion of predictions for each label
        for true_label, predicted_label in zip (Y_dev_fold, Y_dev_pred):
            print ("True Label:", true_label)
            print ("Predicted Label:", predicted_label)
            true_label_name = next ((key for key, value in SENTIMENT_NAME_DIC.items () if value == true_label), None)
            predicted_label_name = next (
                (key for key, value in SENTIMENT_NAME_DIC.items () if value == predicted_label), None)
            if true_label_name is not None and predicted_label_name is not None:
                label_predictions_dev[true_label_name][predicted_label] += 1

        for true_label, predicted_label in zip (Y_test, Y_test_pred):
            true_label_name = next ((key for key, value in SENTIMENT_NAME_DIC.items () if value == true_label), None)
            predicted_label_name = next (
                (key for key, value in SENTIMENT_NAME_DIC.items () if value == predicted_label), None)
            if true_label_name is not None and predicted_label_name is not None:
                label_predictions_test[true_label_name][predicted_label] += 1

        print (f'Fold {fold} Dev Set Predictions:')
        for true_label, predicted_label in zip (Y_dev_fold, Y_dev_pred):
            true_label_name = next (key for key, value in SENTIMENT_NAME_DIC.items () if value == true_label)
            predicted_label_name = next ((key for key, value in SENTIMENT_NAME_DIC.items () if value == predicted_label),None)
            print (f'True Label: {true_label_name}, Predicted Label: {predicted_label_name}')

        print (f'Fold {fold} Test Set Predictions:')
        for true_label, predicted_label in zip (Y_test, Y_test_pred):
            true_label_name = next ((key for key, value in SENTIMENT_NAME_DIC.items () if value == true_label),None)
            predicted_label_name = next((key for key, value in SENTIMENT_NAME_DIC.items () if value == predicted_label),None)
            print (f'True Label: {true_label_name}, Predicted Label: {predicted_label_name}')

    # Calculate and output the prediction ratio for each label
    print ("Dev Set Label Predictions:")
    for label_name, predictions in label_predictions_dev.items ():
        total_predictions = sum (predictions)
        prediction_percentages = [prediction / total_predictions * 100 for prediction in predictions]
        print (
            f"Label: {label_name}, Predicted as Epilepsy: {prediction_percentages[0]:.2f}%, Predicted as FDS: {prediction_percentages[1]:.2f}%")

    print ("Test Set Label Predictions:")
    for label_name, predictions in label_predictions_test.items ():
        total_predictions = sum (predictions)
        prediction_percentages = [prediction / total_predictions * 100 for prediction in predictions]
        print (
            f"Label: {label_name}, Predicted as Epilepsy: {prediction_percentages[0]:.2f}%, Predicted as FDS: {prediction_percentages[1]:.2f}%")
    print(dev_accuracies)
    avg_dev_accuracy = sum(dev_accuracies) / len (dev_accuracies)
    avg_test_accuracy = sum(test_accuracies) / len (test_accuracies)
    avg_test_precision = sum (test_precisions) / len (test_precisions)
    avg_test_recall = sum (test_recalls) / len (test_recalls)
    avg_test_f1_score = sum (test_f1_scores) / len (test_f1_scores)

    print ("Average Dev Set Accuracy:", avg_dev_accuracy)
    print ("Average Test Set Accuracy:", avg_test_accuracy)
    print ("Average Test Set Precision:", avg_test_precision)
    print ("Average Test Set Recall:", avg_test_recall)
    print ("Average Test Set F1-Score:", avg_test_f1_score)


    print ('---------------------------')
