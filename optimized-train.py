# train.py
import os
import pandas as pd
import numpy as np
import joblib
import json
import logging
from typing import Dict, Any, Tuple, List, Optional, Union, Literal
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define model types
ModelType = Literal["regression", "classification"]

@dataclass
class ModelConfig:
    name: str
    model: Any
    model_type: ModelType
    params: Dict[str, Any] = None

# Supported ML models with improved configurations
MODELS = {
    "random_forest_regressor": ModelConfig(
        name="Random Forest Regressor",
        model=RandomForestRegressor(n_estimators=100, random_state=42),
        model_type="regression"
    ),
    "random_forest_classifier": ModelConfig(
        name="Random Forest Classifier",
        model=RandomForestClassifier(n_estimators=100, random_state=42),
        model_type="classification"
    ),
    "ridge": ModelConfig(
        name="Ridge Regression",
        model=Ridge(alpha=1.0, random_state=42),
        model_type="regression"
    ),
    "logistic_regression": ModelConfig(
        name="Logistic Regression",
        model=LogisticRegression(max_iter=1000, random_state=42),
        model_type="classification"
    )
}

# XGBoost is optional since it's not a core scikit-learn dependency
try:
    from xgboost import XGBRegressor, XGBClassifier
    MODELS["xgboost_regressor"] = ModelConfig(
        name="XGBoost Regressor",
        model=XGBRegressor(n_estimators=100, random_state=42),
        model_type="regression"
    )
    MODELS["xgboost_classifier"] = ModelConfig(
        name="XGBoost Classifier",
        model=XGBClassifier(n_estimators=100, random_state=42),
        model_type="classification"
    )
except ImportError:
    logger.warning("XGBoost not installed. XGBoost models will not be available.")

def detect_feature_types(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Detect the types of features in the dataframe and return information about them."""
    feature_info = {}
    
    for col in df.columns:
        # Check if mostly numeric
        numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
        if numeric_count / len(df) > 0.8:  # If more than 80% are valid numbers
            feature_info[col] = {
                "type": "numeric",
                "stats": {
                    "mean": df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None,
                    "min": df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else None,
                    "max": df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else None,
                    "missing": df[col].isna().sum()
                }
            }
        else:
            # Consider categorical
            unique_vals = df[col].nunique()
            feature_info[col] = {
                "type": "categorical",
                "stats": {
                    "unique_values": unique_vals,
                    "top_value": df[col].value_counts().index[0] if not df[col].empty else None,
                    "missing": df[col].isna().sum()
                }
            }
    
    return feature_info

def auto_detect_model_type(target_series: pd.Series) -> ModelType:
    """Automatically detect if this is a classification or regression problem."""
    if pd.api.types.is_numeric_dtype(target_series):
        unique_count = target_series.nunique()
        # If few unique values and they're integers, probably classification
        if unique_count < 10 and all(float(x).is_integer() for x in target_series.dropna().unique()):
            return "classification"
        else:
            return "regression"
    else:
        # Non-numeric target is almost always classification
        return "classification"

def auto_select_algorithm(X: pd.DataFrame, y: pd.Series, model_type: ModelType) -> str:
    """Automatically select an appropriate algorithm based on data characteristics."""
    n_samples, n_features = X.shape
    
    if model_type == "classification":
        if n_samples < 1000:
            return "random_forest_classifier"  # Good for small datasets
        elif "xgboost_classifier" in MODELS and n_samples >= 1000:
            return "xgboost_classifier"  # Better for larger datasets if available
        else:
            return "random_forest_classifier"  # Fallback
    else:  # regression
        if n_samples < 1000:
            return "random_forest_regressor"  # Good for small datasets
        elif "xgboost_regressor" in MODELS and n_samples >= 1000:
            return "xgboost_regressor"  # Better for larger datasets if available
        else:
            return "random_forest_regressor"  # Fallback

def preprocess_data(df: pd.DataFrame, target: str, 
                   feature_info: Optional[Dict[str, Dict[str, Any]]] = None,
                   cat_encoder: str = "onehot") -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer, Dict[str, Dict]]:
    """
    Handles missing values, encoding, and feature scaling with improved handling.
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Basic handling of missing values
    for col in df.columns:
        if df[col].dtype.name == 'object':
            df[col] = df[col].fillna('missing')
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # Extract target
    y = df[target].copy()
    X = df.drop(columns=[target])
    
    # Detect or use provided feature information
    if feature_info is None:
        feature_info = detect_feature_types(X)
    
    # Separate features by type
    cat_features = [col for col, info in feature_info.items() 
                  if info["type"] == "categorical" and col in X.columns]
    num_features = [col for col, info in feature_info.items() 
                  if info["type"] == "numeric" and col in X.columns]
    
    # Create transformers based on feature types
    transformers = [("num", StandardScaler(), num_features)]
    
    if cat_encoder == "onehot":
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        )
    else:  # ordinal
        transformers.append(
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features)
        )
    
    transformer = ColumnTransformer(transformers, remainder='drop')
    
    return X, y, transformer, feature_info

def evaluate_model(y_true, y_pred, model_type, output_dir=None):
    """Evaluate model performance for either regression or classification."""
    if model_type == "regression":
        metrics = {
            "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R²": float(r2_score(y_true, y_pred))
        }
    else:  # classification
        metrics = {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "F1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
    
    # Create visualizations if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if model_type == "regression":
            # Residual plot
            plt.figure(figsize=(10, 6))
            residuals = y_true - y_pred
            plt.scatter(y_pred, residuals)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.savefig(output_dir / 'residual_plot.png')
            plt.close()
        else:
            # Confusion matrix
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig(output_dir / 'confusion_matrix.png')
            plt.close()
    
    return metrics

def train(data_path, target, algorithm=None, model_params=None, output="model_output"):
    """
    ONE-LINER: Train a machine learning model with automatic detection of model type and algorithm.
    
    Args:
        data_path: Path to CSV data file
        target: Name of target column
        algorithm: Algorithm to use (optional, auto-detected if None)
        model_params: Dictionary of model parameters (optional)
        output: Directory to save model and visualizations
        
    Returns:
        Trained model and performance metrics
    """
    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Auto-detect model type
    model_type = auto_detect_model_type(df[target])
    logger.info(f"Auto-detected model type: {model_type}")
    
    # Auto-select algorithm if not specified
    if algorithm is None:
        X_sample = df.drop(columns=[target])
        algorithm = auto_select_algorithm(X_sample, df[target], model_type)
        logger.info(f"Auto-selected algorithm: {algorithm}")
    
    # Validate algorithm
    if algorithm not in MODELS:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose from {list(MODELS.keys())}.")
    
    # Ensure model type matches algorithm
    if MODELS[algorithm].model_type != model_type:
        raise ValueError(
            f"Algorithm '{algorithm}' is for {MODELS[algorithm].model_type} problems, "
            f"but detected a {model_type} problem."
        )
    
    # Preprocess data
    X, y, transformer, feature_info = preprocess_data(df, target)
    joblib.dump(feature_info, output_dir / "feature_info.pkl")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get model
    model_config = MODELS[algorithm]
    model = model_config.model
    
    if model_params:
        model.set_params(**model_params)
    
    # Create and train pipeline
    pipeline = Pipeline([
        ("transform", transformer),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate model
    metrics = evaluate_model(y_test, y_pred, model_type, output_dir)
    
    # Save model performance
    performance_data = {
        "Model": MODELS[algorithm].name,
        "Type": model_type,
        "Algorithm": algorithm,
        **metrics
    }
    
    with open(output_dir / "model_performance.json", "w") as f:
        json.dump(performance_data, f)
    
    # Save model
    model_path = output_dir / "model.pkl"
    joblib.dump(pipeline, model_path)
    
    # Create symlinks for API to find
    if os.path.exists("model.pkl"):
        os.remove("model.pkl")
    if os.path.exists("feature_info.pkl"):
        os.remove("feature_info.pkl")
    
    try:
        os.symlink(os.path.abspath(model_path), "model.pkl")
        os.symlink(os.path.abspath(output_dir / "feature_info.pkl"), "feature_info.pkl")
    except Exception as e:
        logger.warning(f"Could not create symlinks: {str(e)}. Copying files instead.")
        import shutil
        shutil.copy(model_path, "model.pkl")
        shutil.copy(output_dir / "feature_info.pkl", "feature_info.pkl")
    
    # Print summary
    primary_metric = "RMSE" if model_type == "regression" else "Accuracy"
    primary_metric_value = metrics.get(primary_metric)
    print(f"\nModel Training Summary:")
    print(f"- Algorithm: {MODELS[algorithm].name}")
    print(f"- Model Type: {model_type}")
    print(f"- {primary_metric}: {primary_metric_value:.4f}")
    print(f"- Model saved to: {os.path.abspath(model_path)}")
    print("\nTo deploy the model API, run: onelinerml.deploy()")
    
    return pipeline, performance_data

# FastAPI app definition (moved from api.py to here for simplicity)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model at startup
    try:
        app.state.model = joblib.load("model.pkl")
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error("Model file not found. Make sure to train a model first.")
        app.state.model = None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        app.state.model = None
    
    yield
    
    # Clean up resources at shutdown
    app.state.model = None
    logger.info("API shutting down, resources cleaned up")

def deploy(host="127.0.0.1", port=8000):
    """
    ONE-LINER: Deploy a trained model as a FastAPI web service.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
    """
    app = FastAPI(
        title="OnelinerML API",
        description="API for machine learning model predictions",
        version="1.0.0",
        lifespan=lifespan
    )
    
    class HealthResponse(BaseModel):
        status: str
        model_info: Dict[str, Any]

    # Define endpoints
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        model_info = {"model_type": type(app.state.model).__name__}
        
        # Add performance metrics if available
        try:
            with open("model_performance.json", "r") as f:
                performance = json.load(f)
            model_info.update(performance)
        except Exception:
            pass
            
        return HealthResponse(status="ok", model_info=model_info)
    
    @app.get("/predict/")
    async def predict(data: Dict[str, Any]):
        """Serve predictions via GET endpoint."""
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
            
        try:
            df = pd.DataFrame([data])
            prediction = app.state.model.predict(df)
            result = {"prediction": prediction[0]}
            
            # Add probabilities for classification models
            if hasattr(app.state.model, "predict_proba"):
                try:
                    proba = app.state.model.predict_proba(df)[0]
                    if hasattr(app.state.model, "classes_"):
                        result["prediction_probability"] = {
                            str(cls): float(prob) for cls, prob in zip(app.state.model.classes_, proba)
                        }
                except Exception:
                    pass
                    
            return result
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    print(f"API running at: http://{host}:{port}/predict")
    print(f"Documentation at: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)
