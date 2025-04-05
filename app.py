# Лабораторная работа №3: Разведочный анализ данных
# Цель: Анализ предобработанного датасета Airbnb_Open_Data.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 1. Загрузка данных
df = pd.read_csv('Airbnb_Open_Data.csv', encoding='utf-8')

# Предобработка: убираем символы валют и преобразуем нужные столбцы
df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
df['number of reviews'] = pd.to_numeric(df['number of reviews'], errors='coerce')

# Удалим строки с пропущенными значениями в важных колонках
df = df.dropna(subset=['price', 'number of reviews', 'room type'])

# 2. Первые 5 строк
print("Первые 5 строк датасета:")
print(df.head())

# ✅ 3. Замена одного значения на моду
room_mode = df['room type'].mode()[0]
df.iloc[0, df.columns.get_loc('room type')] = room_mode
print(f"\nПервое значение в 'room type' заменено на моду: {room_mode}")

# ✅ 4. Кодирование категориальной переменной
le = LabelEncoder()
df['room type encoded'] = le.fit_transform(df['room type'])
print("\nПервые 5 строк после кодирования:")
print(df[['room type', 'room type encoded']].head())

# 5. Распределение переменной "price"
plt.hist(df['price'], bins=30, color='black', alpha=0.8)
plt.xlabel('Цена')
plt.ylabel('Частота')
plt.title('Распределение цены')
plt.show()

# 6. Корреляция между переменными
correlation = df['price'].corr(df['number of reviews'])
print(f"\nКоэффициент корреляции между ценой и числом отзывов: {correlation:.2f}")

# 7. Анализ выбросов (boxplot по цене)
plt.boxplot(df['price'])
plt.ylabel('Цена')
plt.title('Анализ выбросов цены')
plt.show()

# 8. Частоты категориальных переменных: room type
room_counts = df['room type'].value_counts()
room_counts.plot(kind='bar', color='gray')
plt.xlabel('Тип комнаты')
plt.ylabel('Количество')
plt.title('Распределение типов комнат')
plt.show()

# 9. Общий дешборд
plt.figure(figsize=(10, 8))

# Подграфик 1: гистограмма
plt.subplot(2, 2, 1)
plt.hist(df['price'], bins=30, color='black', alpha=0.8)
plt.xlabel('Цена')
plt.ylabel('Частота')
plt.title('Распределение цены')

# Подграфик 2: scatter
plt.subplot(2, 2, 2)
plt.scatter(df['price'], df['number of reviews'], color='black', alpha=0.5)
plt.xlabel('Цена')
plt.ylabel('Число отзывов')
plt.title('Диаграмма рассеяния')

# Подграфик 3: boxplot
plt.subplot(2, 2, 3)
plt.boxplot(df['price'])
plt.ylabel('Цена')
plt.title('Выбросы')

# Подграфик 4: тепловая карта корреляции
plt.subplot(2, 2, 4)
corr_matrix = df[['price', 'number of reviews']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2g', cmap='Blues')
plt.title('Карта корреляций')

plt.tight_layout()
plt.show()
