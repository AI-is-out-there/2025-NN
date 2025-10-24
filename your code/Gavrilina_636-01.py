# Подключение библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Работа с датасетом. Так как на данный момент у меня нет готового, данные генерируются рандомным образом с некоторыми ограничениями.
# Факторы: возраст, D-димер (нг/мл), иммобилизация (0/1), хирургия (0/1), рак (0/1), гипотензия (0/1, риск кровотечения), недавнее кровотечение (0/1), HAS-BLED score (риск кровотечения, 0-9)

np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    'age': np.random.normal(65, 15, n_samples).clip(18, 100),
    'd_dimer': np.random.exponential(500, n_samples).clip(0, 5000), 
    'immobilization': np.random.binomial(1, 0.3, n_samples),
    'surgery': np.random.binomial(1, 0.2, n_samples),
    'cancer': np.random.binomial(1, 0.15, n_samples),
    'hypotension': np.random.binomial(1, 0.1, n_samples), 
    'recent_bleed': np.random.binomial(1, 0.08, n_samples),
    'has_bled': np.random.poisson(1.5, n_samples).clip(0, 9)  
})

# Целевая переменная: 1 если ТЭЛА вероятно И низкий риск кровотечения (нужна терапия), 0 иначе
thrombo_risk = 0.5 * (data['d_dimer'] > 500) + 0.2 * data['immobilization'] + 0.15 * data['surgery'] + 0.1 * data['cancer'] + 0.05 * (data['age'] > 70)
bleed_risk = 0.3 * data['hypotension'] + 0.25 * data['recent_bleed'] + 0.1 * data['has_bled'] / 3
y = (thrombo_risk > 0.4) & (bleed_risk < 0.3)  # Благоприятный исход для терапии
y = y.astype(int)
X = data.values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создала на данном этапе упрощенную нейронную сеть
model = Sequential([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),  # Входной слой
    Dropout(0.2),  # Регуляризация
    Dense(8, activation='relu'),  # Скрытый слой 1
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Выход: вероятность благоприятного исхода
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Процесс обучения
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Для класса 0 (Не рекомендовать терапию):
# Precision = TN / (TN + FP) — доля правильных решений "не лечить" среди всех решений "не лечить"
# Recall = TN / (TN + FN) — доля правильных "не лечить" среди всех случаев, где терапия действительно не нужна
# Для класса 1 (Рекомендовать терапию):
# Precision = TP / (TP + FP) — доля правильных рекомендаций среди всех рекомендаций
# Recall = TP / (TP + FN) — доля корректно выявленных пациентов, нуждающихся в терапии
# F1-score = 2 × (Precision × Recall) / (Precision + Recall) — гармоническое среднее, балансирует precision и recall

y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nОтчет по классификации:")
print(classification_report(y_test, y_pred))
