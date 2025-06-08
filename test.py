# test.py
import torch
from model import InverseModel
from data_processing import load_and_process_data
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import pandas as pd

def test_model(data_path, model_path="checkpoints/inverse_model.pt", batch_size=256, device='cpu'):
    # 1. Veriyi hazırla
    X_train, X_test, y_train, y_test, x_scaler, y_scalers, component_cols = load_and_process_data(data_path)

    # 2. Modeli oluştur ve yükle
    model = InverseModel(input_dim=X_test.shape[1], output_dim=y_test.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Test verisini tensöre çevir
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy()
        targets = y_test_tensor.cpu().numpy()

    # 4. Komponent bazlı metrikleri hesapla
    results = []
    print("\n=== Component Bazlı MAE, R² ve % Hata ===")
    for i, col in enumerate(component_cols):
        # Normal formata çevir
        pred_inv = y_scalers[col].inverse_transform(preds[:, i].reshape(-1, 1)).flatten()
        true_inv = y_scalers[col].inverse_transform(targets[:, i].reshape(-1, 1)).flatten()

        mae = mean_absolute_error(true_inv, pred_inv)
        r2 = r2_score(true_inv, pred_inv)
        mean_true = np.mean(np.abs(true_inv))
        percent_error = (mae / mean_true) * 100 if mean_true > 1e-12 else 0.0

        results.append((col, mae, r2, percent_error))
        print(f"{col:>10} | MAE: {mae:>12.6f} | R² Score: {r2:>7.5f} | % Hata: {percent_error:6.2f}%")

    mean_r2 = np.mean([r[2] for r in results])
    print(f"\n📈 Ortalama R² Skoru: {mean_r2:.5f}")

    # 5. CSV olarak kaydet
    df_results = pd.DataFrame(results, columns=["Component", "MAE", "R2", "Percent_Error"])
    df_results.to_csv("test_metrics.csv", index=False)
    print("✅ test_metrics.csv dosyasına kaydedildi.")

if __name__ == "__main__":
    test_model("C:/Users/gokha/Desktop/MODEL V8/filtered_data_clean.csv")
