from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(model, X_test, y_test, task_type):
    y_pred = model.predict(X_test)
    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'F1 Score': f1_score(y_test, y_pred, average='macro')
    }
    if task_type == 'Classification':
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            results['AUC'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
        results['Confusion Matrix'] = confusion_matrix(y_test, y_pred).tolist()
    return results
