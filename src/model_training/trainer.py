import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

MODELS = {
    'Linear Regression': LinearRegression,
    'Logistic Regression': LogisticRegression,
    'Decision Trees': DecisionTreeClassifier,
    'Random Forest': RandomForestClassifier,
    'Support Vector Machine': SVC,
    'Naive Bayes': GaussianNB,
    'KNN': KNeighborsClassifier
}

def train_model(model_name, X_train, y_train):
    model_cls = MODELS.get(model_name)
    if not model_cls:
        raise ValueError(f"Unsupported model: {model_name}")
    model = model_cls()
    model.fit(X_train, y_train)
    return model