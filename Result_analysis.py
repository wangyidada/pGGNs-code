import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score


def AUC_Confidence_Interval(y_true, y_pred, CI_index=0.95):
    AUC = roc_auc_score(y_true, y_pred)

    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int((1.0 - CI_index) / 2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(1.0 - (1.0 - CI_index) / 2 * len(sorted_scores))]
    CI = [confidence_lower, confidence_upper]

    print('AUC is {:.3f}, Confidence interval : [{:0.3f} - {:0.3}]'.format(AUC, confidence_lower, confidence_upper))
    return AUC, CI, sorted_scores


def cal_auc(label, score):
    [AUC, CI, sorted_scores] = AUC_Confidence_Interval(label, score, CI_index=0.95)
    fpr, tpr, thresholds = roc_curve(label, score)
    return AUC, fpr, tpr, thresholds


def plot_roc(train_AUC, train_fpr, train_tpr,
             val_AUC, val_fpr, val_tpr,
             test_AUC, test_fpr, test_tpr):
    lw = 1.5
    plt.figure(figsize=(5, 5))
    plt.plot(train_fpr, train_tpr, color='green', lw=lw, label='Training (AUC = %0.2f)' % train_AUC)
    plt.plot(val_fpr, val_tpr, color='blue', lw=lw, label='Validation (AUC = %0.2f)' % val_AUC)
    plt.plot(test_fpr, test_tpr, color='red', lw=lw, label='Testing ROC (AUC = %0.2f)' % test_AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc="lower right")
    plt.title('ROC')
    plt.savefig('ROC.png', dpi = 300)
    plt.show()


def evaluation(fpr, tpr, t, score, label):
    RightIndex = (tpr + (1 - fpr) - 1)
    index = np.where(RightIndex == np.max(RightIndex))[0]
    threshold = t[index]
    for z in range(len(index)):
        test_threshold = t[index[z]]
        score[np.where(score < threshold)] = 0
        score[np.where(score >= threshold)] = 1
        right_index = np.where(score == label)[0]
        wrong_index = np.where(score != label)[0]
        ACC = len(right_index) / len(label)

        TP = len(np.where(label[right_index] == 1)[0])
        TN = len(np.where(label[right_index] == 0)[0])
        FP = len(np.where(label == 0)[0]) - TN
        FN = len(np.where(label == 1)[0]) - TP

        Sensitivity = TP / (TP + FN)
        Specificiry = TN / (FP + TN)

        print('accuracy is %0.2f, Sensitivity is %0.2f, Specificiry is %0.2f, threshold is %0.2f'
              % (ACC, Sensitivity, Specificiry, test_threshold))
        print('TP is %0.2f, TN is %0.2f,FP is %0.2f,FN is %0.2f ' % (TP, TN, FP, FN))
        # print('wrong-case is %d ' % wrong_index)


def classificaton_evaluation(values):
    # values = [[train_label, train_score], [val_label, val_score], [test_label, test_score]]
    train_score = np.asarray(values[0][0], dtype=np.float)
    train_label = np.asarray(values[0][1], dtype=np.int)
    val_score = np.asarray(values[1][0], dtype=np.float)
    val_label = np.asarray(values[1][1], dtype=np.int)
    test_score = np.asarray(values[2][0], dtype=np.float)
    test_label = np.asarray(values[2][1], dtype=np.int)

    [train_AUC, train_fpr, train_tpr, train_thresholds] = cal_auc(train_label, train_score)
    [val_AUC, val_fpr, val_tpr, val_thresholds] = cal_auc(val_label, val_score)
    [test_AUC, test_fpr, test_tpr, test_thresholds] = cal_auc(test_label, test_score)
    plot_roc(train_AUC, train_fpr, train_tpr,val_AUC, val_fpr, val_tpr,test_AUC, test_fpr, test_tpr)

    print('Trian Dataset result:\n')
    evaluation(train_fpr, train_tpr, train_thresholds, train_score, train_label)
    print('Val Dataset result: \n')
    evaluation(val_fpr, val_tpr, val_thresholds, val_score, val_label)
    print('Test Dataset result: \n')
    evaluation(test_fpr, test_tpr, test_thresholds, test_score, test_label)


def plot_roc_curve(values, colors=['blue', 'green', 'purple']):
    # values = [[train_label, train_score], [val_label, val_score], [test_label, test_score]]
    for i in range(len(values)):
        labels = np.asarray(values[i][1], dtype=np.int)
        logits = np.asarray(values[i][0], dtype=np.float)
        FPR, TPR, t = roc_curve(labels, logits)
        roc_auc = auc(FPR, TPR)
        plt.plot(FPR, TPR, colors[i], label='AUC=%0.4f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig('ROC.png', dpi=500)
    plt.show()


