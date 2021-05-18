
def get_fnr(y_true, y_pred, thresh, label):
    return (y_pred[y_true == label] < thresh).sum() / float((y_true == label).sum())


def get_fpr(y_true, y_pred):
    nbenign = (y_true == 0).sum()
    nfalse = (y_pred[y_true == 0] == 1).sum()
    return nfalse / float(nbenign)


def get_threshold(y_true, y_pred, fpr_target):
    thresh = 0.0
    fpr = get_fpr(y_true, y_pred > thresh)
    while fpr > fpr_target and thresh < 1.0:
        thresh += 0.0001
        fpr = get_fpr(y_true, y_pred > thresh)
    return thresh, fpr


def get_detection_rate(fnr):
    return 100 - fnr


def plot_roc(y_true, y_pred, fpr, fnr):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_curve
    plt.figure(figsize=(8, 8))
    fpr_plot, tpr_plot, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr_plot, tpr_plot, lw=4, color='k')
    plt.gca().set_xscale("log")
    plt.yticks(np.arange(22) / 20.0)
    plt.xlim([4e-5, 1.0])
    plt.ylim([0.65, 1.01])
    plt.gca().grid(True)
    plt.vlines(fpr, 0, 1 - fnr, color="r", lw=2)
    plt.hlines(1 - fnr, 0, fpr, color="r", lw=2)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    _ = plt.title("Ember Model ROC Curve")
    return _.figure
