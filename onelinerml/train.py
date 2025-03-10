import pandas as pd
from sklearn.model_selection import train_test_split
from .preprocessing import preprocess_data
from .models import get_model
from .evaluation import evaluate_model

def train(data_source, model="linear_regression", target_column="target", test_size=0.2, random_state=42, **kwargs):
    if isinstance(data_source, str):
        data = pd.read_csv(data_source)
    else:
        data = data_source
    
    X, y = preprocess_data(data, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    model_instance = get_model(model, **kwargs)
    model_instance.fit(X_train, y_train)
    
    metrics = evaluate_model(model_instance, X_test, y_test)
    return model_instance, metrics
