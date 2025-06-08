# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import InverseModel
from data_processing import load_and_process_data
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import os

def train_model(data_path, epochs=75, batch_size=256, lr=2e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Veriyi yükle ve ölçekle
    X_train, X_test, y_train, y_test, x_scaler, y_scalers, component_cols = load_and_process_data(data_path)

    # PyTorch veri yapıları
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model ve optimizer
    model = InverseModel(input_dim=X_train.shape[1], output_dim=y_train.shape[1]).to(device)
    criterion = nn.SmoothL1Loss(reduction='none')  # loss'u her output için hesaplayacağız
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Komponent ağırlıkları (örnek: cs, c1, c2 daha ağır)
    component_weights = torch.tensor([1.5, 2.0, 2.0, 1.5, 3.5, 7.0, 2.5, 2.5, 2.5, 7.0], dtype=torch.float32).to(device)
    component_weights /= component_weights.sum()

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)

            # Her output için loss + ağırlıklı ortalama
            loss_raw = criterion(preds, batch_y)  # (B, 10)
            loss = (loss_raw * component_weights).sum(dim=1).mean()  # toplam loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Değerlendirme
        model.eval()
        val_losses = []
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss_raw = criterion(preds, batch_y)
                loss = (loss_raw * component_weights).sum(dim=1).mean()
                val_losses.append(loss.item())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        maes, r2s = [], []
        for i, col in enumerate(component_cols):
            y_true = y_scalers[col].inverse_transform(all_targets[:, i].reshape(-1, 1)).flatten()
            y_pred = y_scalers[col].inverse_transform(all_preds[:, i].reshape(-1, 1)).flatten()
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            maes.append(mae)
            r2s.append(r2)

        weighted_mae = np.sum(np.array(maes) * component_weights.cpu().numpy())

        print(f"\nEpoch {epoch+1:03d}: "
              f"Val Loss: {np.mean(val_losses):.4f} | "
              f"Train Loss: {np.mean(train_losses):.4f} | "
              f"Mean MAE: {np.mean(maes):.4f} | "
              f"Weighted MAE: {weighted_mae:.4f} | "
              f"Mean R²: {np.mean(r2s):.5f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/inverse_model.pt")
    print("\n✅ Model eğitildi ve kaydedildi: checkpoints/inverse_model.pt")

if __name__ == "__main__":
    train_model("C:/Users/gokha/Desktop/MODEL V8/filtered_data_clean.csv")
