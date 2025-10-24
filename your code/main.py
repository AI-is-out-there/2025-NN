import optuna
import torch
import numpy as np
import sklearn
import torchmetrics
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.nn as nn
from sklearn.datasets import fetch_covtype
torch.manual_seed(42)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
elif torch.cpu.is_available():
    device = 'cpu'

# Используем изученные модели нейронных сетей и тренеровки

def train2(model, optimizer, criterion, metric, train_loader, valid_loader,
               n_epochs):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        for X_batch, y_batch in train_loader:
            model.train()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
    return history

def evaluate_tm(model, data_loader, metric):
    model.eval()
    metric.reset()  # reset the metric at the beginning
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()

# Нейронная сеть стандартной двуслойной конфигурации. Используем один выход
# дабы предсказывать количество дней.

class DayPredictor(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, 1)
        )

    def forward(self, X):
        return self.mlp(X)





# ---------- Данные и предобработка ----------

# Загружем данные и предобрабатываем их.

df = pd.read_csv('dirty_v3_path.csv')

# Так как большинство пациентов наверняка будут иметь пропущенные данные,
# то приходитсья дополнять клетки, так как иначе мы не сможем помочь много кому.
df = df.fillna(df.mean(numeric_only=True))

# Отделяем нужные данны и выбрасывам нерелевантные данные.
y = df['LengthOfStay'].values
X = df.drop(columns=['LengthOfStay', 'random_notes', 'noise_col'])

# Некоторые поля имеют текстовый тип, поэтому переводим их в через классификатор
# в числовой тип.
gender_encoder = LabelEncoder()
X['Gender'] = gender_encoder.fit_transform(X['Gender'])

medical_encoder = LabelEncoder()
X['Medical Condition'] = medical_encoder.fit_transform(X['Medical Condition'])

# Разделяем данные на несколько выборок для обучения и тестирования
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.35, random_state=42)

# Нормализуем данные для повышения точности работы модели.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
valid_dataset = TensorDataset(torch.FloatTensor(X_valid), torch.FloatTensor(y_valid).unsqueeze(1))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)





# ---------- Подбор параметров нейронной сети ----------

# Используем среднюю абсолютную ошибку как меритель, так как наша задача максимально
# близко предугадать время.
accuracy = torchmetrics.MeanAbsoluteError().to(device)
# Используем MSELoss как сдандарт для простой задачи.
loss = nn.MSELoss()

print("##### Looking for best parameters ######")

# Подбираем наиболее оптимальные параметры. Ищем что-то, что может предсказывать
# со средней ошибкой около 1.3, или лучший из 20 попыток.
def objective(trial):
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_hidden1 = trial.suggest_int("n_hidden1", 8, 256)
    n_hidden2 = trial.suggest_int("n_hidden2", 8, 256)
    model = DayPredictor(X_train.shape[1], n_hidden1, n_hidden2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = train2(model, optimizer, loss, accuracy, train_loader, valid_loader, 10)
    validation_accuracy = min(history["valid_metrics"])
    return validation_accuracy

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)

def stop_condition(study, trial):
  if study.best_value <= 1.3:
      study.stop()

study.optimize(objective, callbacks=[stop_condition], n_trials=2)





# ---------- Обучение ----------

print("\n\n\n\n##### Learning the model with best paramters ######")
print(f'First layer: {study.best_params['n_hidden1']}, Second layer: {study.best_params['n_hidden2']}\n')

# Обучаем модель по найденным параметрам
model = DayPredictor(n_inputs=X_train.shape[1],
                     n_hidden1=study.best_params['n_hidden1'],
                     n_hidden2=study.best_params['n_hidden2']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=study.best_params['lr'])
# Используем оптимизатор Adam так как он показал лучшие резульаты по точности
# и скорости сходимости с другими тестируемыми алгоритмами.
_ = train2(model, optimizer, loss, accuracy, train_loader, valid_loader,
           n_epochs=20)





# ---------- Тест точности ----------

print("\n\n\n\n##### Running tests to see accuracy ######")

# Выводим несколько пробных запросов дабы дать пример точности
model.eval()
with torch.no_grad():
    indices = torch.randperm(len(test_dataset))[:5]

    for idx in indices:
        X_sample, y_actual = test_dataset[idx]
        X_sample = X_sample.to(device)
        y_pred = model(X_sample)
        print(f"Prediction: {y_pred.item() :.1f} Actual: {y_actual.item() :.1f}")
