# data_processing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_process_data(path):
    df = pd.read_csv(path)

    # Gain sütunlarını ve komponent sütunlarını ayır
    gain_cols = [col for col in df.columns if "gain" in col.lower()]
    component_cols = ['rin', 'rg1', 'rg2', 'rd', 'rs', 'cs', 'c1', 'c2', 'rl', 'vdd']

    # Giriş (X) ve hedef (y) verilerini ayır
    X = df[gain_cols].values
    y = df[component_cols].copy()

    # Girişleri standardize et
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)

    # Komponentleri log + MinMax ile ölçekle
    y_scaled = y.copy()
    y_scalers = {}

    for col in component_cols:
        scaler = MinMaxScaler()

        if col.startswith('r') or col.startswith('c'):
            if col in ['c1', 'c2']:
                y_scaled[col] = np.log10(y[col] + 1e-11)
            else:
                y_scaled[col] = np.log1p(y[col])  # log(1 + x)

        y_scaled[col] = scaler.fit_transform(y_scaled[[col]])
        y_scalers[col] = scaler

    # Eğitim / test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled.values, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, x_scaler, y_scalers, component_cols
