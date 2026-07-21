import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

def unpack_active_data(active_dict: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    X_train = active_dict["train"]["window_matrix"]
    y_train = active_dict["train"]["answers"]
    
    X_test = active_dict["test"]["window_matrix"]
    y_test = active_dict["test"]["answers"]
    
    scaler = active_dict["MinMaxScaler"]
    
    return X_train, y_train, X_test, y_test, scaler


def train_svr(X_train: np.ndarray, y_train: np.ndarray, kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1) -> SVR:

    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_train, y_train)
    return model


def predict_svr(model: SVR, X_test: np.ndarray) -> np.ndarray:
    predictions_scaled = model.predict(X_test)
    return predictions_scaled

def denormalize_data(scaled_array: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    reshaped_array = scaled_array.reshape(-1, 1)

    denormalized_array = scaler.inverse_transform(reshaped_array).flatten()
    
    return denormalized_array

def run_svr_pipeline(active_dict: dict, kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1) -> dict:

    X_train, y_train, X_test, y_test, scaler = unpack_active_data(active_dict)

    model = train_svr(X_train, y_train, kernel=kernel, C=C, epsilon=epsilon)

    predictions_scaled = predict_svr(model, X_test)

    predictions_real = denormalize_data(predictions_scaled, scaler)
    answers_real = denormalize_data(y_test, scaler)

    return {
        "predicted": predictions_real,
        "answers": answers_real
    }