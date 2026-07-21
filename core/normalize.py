import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_active_data(active_dict: dict) -> dict:
    train_raw = active_dict["train"]
    test_raw = active_dict["test"]

    scaler = MinMaxScaler(feature_range=(0, 1))

    train_reshaped = train_raw.reshape(-1, 1)
    train_scaled = scaler.fit_transform(train_reshaped).flatten()

    test_reshaped = test_raw.reshape(-1, 1)
    test_scaled = scaler.transform(test_reshaped).flatten()
    
    active_dict["train"] = train_scaled
    active_dict["test"] = test_scaled
    active_dict["MinMaxScaler"] = scaler
    
    return active_dict