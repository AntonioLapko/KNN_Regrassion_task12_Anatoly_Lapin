from sklearn import datasets
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from mlxtend.plotting import plot_decision_regions

dataset_diabetes = pd.read_csv('diabetes.csv')

#Приведение к нижнему регистру
dataset_diabetes.columns = [col.lower() for col in dataset_diabetes.columns]

print(dataset_diabetes.head(), "\n")
print(dataset_diabetes.info(), "\n")

rows, cols = dataset_diabetes.shape
print(f"В исходном датасете {rows} строк", "\n")
print(f"В исходном датасете {cols} столбцов", "\n")

#Проведите EDA(Exploratory Data Analysis)
print(f"Доля пропусков в исходном датасете")
print(dataset_diabetes.isna().mean().sort_values(), "\n")

print(f"Максимальные значения в исходном датасете")
print(dataset_diabetes.max(), "\n")

print(f"Минимальные значения в исходном датасете")
print(dataset_diabetes.min(), "\n")

print(f"Средние значения в исходном датасете")
print(dataset_diabetes.mean(), "\n")

print(f"Медиана исходного датасета")
print(dataset_diabetes.median(), "\n")

print(f"Дисперсия")
print(dataset_diabetes.var(), "\n")

print(f"Среднеквадратическое отклонение")
print(dataset_diabetes.std(), "\n")

# Создание списка значений k, которые оцениваем
k_values = [num for num in range(5, 26, 5)]
print(k_values)

# Инициализация списка для сохранения средних оценок точности для каждого значения k
mean_scores = []

X = dataset_diabetes.drop('outcome', axis=1)
y = dataset_diabetes['outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1234)


# Перебор значений k
for k in k_values:
    # Создание модели k-ближайших соседей
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Вычисление средней оценки точности с использованием перекрестной проверки
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')  # cv=5 означает 5-кратную перекрестную проверку
    mean_scores.append(np.mean(scores))


# Вывод результатов
for k, score in zip(k_values, mean_scores):
    print (f"k ={k} Mean Accuracy: {score}")

