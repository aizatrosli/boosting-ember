from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, make_scorer

'''
Scoring methodology return values as dictionary
eg. 
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
'''