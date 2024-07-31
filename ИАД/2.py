import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

# Просмотр данных
print(df)


# Характеристики датасета - описание
print(df.describe())
print(df.info())
df.rename(columns = {'Price':'Цена', 'Apartment type':'Тип квартиры', 'Metro station':'Станция метро',
                     'Minutes to metro':'Минуты до метро', 'Region':'Регион', 'Number of rooms':'Число комнат',
                     'Area':'Площадь', 'Living area':'Жилая площадь', 'Kitchen area':'Площадь кухни', 'Floor':'Этаж',
                     'Number of floors':'Число этажей', 'Renovation':'Ремонт'}, inplace = True)
print(df.head())


# Работа с отдельными столбцами или строками
print("Выведем цену, тип и площадь")
print(df.loc[:, ['Цена', 'Тип квартиры', 'Площадь']])

print("Вывод 101 первых строк и 6 столбцов")
print(df.iloc[0:100, 0:5])



# Использовать Numpy для датасета
mean_Area = np.mean(df['Площадь'])
print('Средняя площадь квартиры:', mean_Area)


# Использовать SciPy для датасета
x = np.arange(-200, 350, 1)
mu, std = norm.fit(df['Площадь'])
p = norm.pdf(x, mu, std)
plt.plot(x, p, linewidth=2)
plt.annotate('MEAN', xy=(mean_Area, 0),  xycoords='data',
            xytext=(mean_Area, 0.003), textcoords='data',
            arrowprops=dict(facecolor='g'))
plt.xlabel("Площадь")
plt.show()

# Использовать метод tolist()
volumes = df['Цена'].tolist()
print("Вывод значения Цены")
print(volumes)

cols=df.columns.tolist()
print("Вывод названия колонок")
print(cols)


# Применить циклы для любого вида анализа или преобразований
# Добавление новых строк и столбцов
df['Цена за метр'] = df['Цена'] / df['Площадь']
df_new = df.head()
print("Индекс    Цена за метр", df['Цена за метр'], sep="\n")

for index in range(len(df)):
    if index % 50 == 0:
        new_row = df.iloc[index]
        df_new = pd.concat([df_new, new_row.to_frame().T], ignore_index=True)
print("df_new")
print(df_new)

# Удаление строк и столбцов
df.drop(labels='Цена за метр', axis = 1, inplace = True)
print(df)

# Поиск Уникальные значения
print("Уникальные значения ремонта:", df.Ремонт.unique())

# Группировка данных для анализа Pandas
df_grouped=df.groupby('Ремонт', as_index=False)[['Цена', 'Площадь']].mean()
df_grouped = df_grouped.rename(columns={'Цена': 'Средняя цена', 'Площадь':'Средняя площадь'})
print(df_grouped)

# Сводные таблицы
print("Сводная")
print(pd.pivot_table(df, index = ['Ремонт'], columns = ['Тип квартиры'], values = 'Цена', aggfunc = 'mean', margins=True))

# Сортировка данных
df['Цена за метр'] = df['Цена'] / df['Площадь']
print(df.sort_values(by='Цена за метр', ascending=True))


# Фильтрация
print("Профильтруем")
print(df[(df['Цена за метр'] < 100000) & (df['Площадь'] > 50 )])

# Применение функций к столбцам
def my_lower(row): return row.lower()
print('печатаем тип с маленькими буквами')
print(df['Тип квартиры'].apply(my_lower))


# Очистка данных
df_copied = df.copy()
rows_to_drop = df_copied[(df_copied['Цена за метр'] > 100000) | (df_copied['Площадь'] < 50)].index
print('Индексы на удаление:', rows_to_drop)
df_copied.drop(rows_to_drop, inplace=True)
print(df_copied)

# Построение графиков (использовать Mathplotlib, Plotly)
plt.scatter(df['Площадь'], df['Цена'], linewidth=2)
plt.xlabel("Площадь")
plt.ylabel("Цена")
plt.title('Площадь объекта и цена')
plt.show()

plt.hist(df['Число комнат'], bins=np.arange(0.5, 6.5, 1), edgecolor='black', linewidth=1.2)
plt.xlabel('Число комнат')
plt.ylabel('Число объектов')
plt.title('Гистограмма распределения количества комнат')
plt.show()

df_grouped=df.groupby('Площадь', as_index=False)[['Цена']].mean()
df_grouped.sort_values(by='Площадь', ascending=True)
df_grouped.plot(x = 'Площадь', y = 'Цена')
plt.title('График Цены от площади квартиры')
plt.show()

