import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
def Life(a):
    if a==0:
        return "Не выживет"
    else:
        return "Выживет"
# Интерпретация данных
# PassengerId: Уникальный индекс/номер строки. Начинается с 1 (для первой строки) и увеличивается на 1 для каждой следующей.
# Рассматриваем его как индентификатор строки и, что логично, идентификатор пассжира (т.к. для каждого пассажира в датасете представлена только одна строка).
# Survived: Признак, показывающий был ли спасен данный пассажир или нет. 1 означает, что удалось выжить, и 0 - не удалось спастись.
# Pclass: Класс билета. 1 - означает Первый класс билета. 2 - означает Второй класс билета. 3 - означает Третий класс билета.
# Name: Имя пассажира. Имя также может содержать титулы и обращения. "Mr" для мужчин. "Mrs" для женщин. "Miss" для девушек
# (тут имеется в виду что для тех, кто не замужем, так было принято, да и сейчас тоже, говорить в западном обществе). "Master" для юношей.
# Sex: Пол пассажира. Либо мужчины (=Male) либо женщины (=Female).
# Age: Возраст пассажира. "NaN" значения в этой колонке означают, что возраст данного пассажира отсутствует/неизвестен/или не был записанv в датасет.
# SibSp: Количество братьев/сестер или супругов, путешествующих с каждым пассажиром.
# Parch: Количество родителей детей (Number of parents of children travelling with each passenger).
# Ticket: Номер билета.
# Fare: Сумма, которую заплатил пассажир за путешествие.
# Cabin: Номер каюты пассажира. "NaN" значения в этой колонке указавает на то, что номер каюты данного пассажира не был записан.
# Embarked: Порт отправления данного пассажира.

# Для отображения всех столбцов
pd.set_option('display.max_columns', None)


train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')

print('Кол-во строчек и столбцов', train.shape)

# отобразит различные величины, такие как количесmво, среднее,
# среднеквадратичное отклонение и т.д. для численных типов данных.
print('Различные показатели')
print(train.describe())

#  отобразит статистики (descriptive statistics) объектного типа.
#  Это нужно для нечисловых данных, когда нельзя просто посчитать максимумы/среднее/и пр.
#  для данных. Мы можем отнести такие данные к категориальному виду.
print('\nРазличные показатели для не числовых показателей')
print(train.describe(include=['O']))

# Больше информации о типах данных/структуре в тренировочной выборке
# Можно увидеть, что значение Age не задано для большого количества записей.
# Из 891 строк, возраст Age задан лишь для 714 записей.
# Аналогично, Каюты Cabin также пропущены для большого количества записей.
# Только 204 из 891 записей содержат Cabin значения.
print('\nИнформация о типах данных и null значениях')
print(train.info())

# Всего 177 записей с пропущенным возрастом (Age),
# 687 записей с пропущенным значение каюты Cabin
# и для 2 записей не заданы порты отправления Embarked.
print('\nNone значения')
print(train.isnull().sum())
train = train.dropna(subset=['Age'])
print('\nNone значения')
print(train.isnull().sum())

print('\nNone значения в тестовой')
print(test.isnull().sum())
test = test.dropna(subset=['Age'])
print('\nNone значения')
print(test.isnull().sum())


columns_target=["Survived"] #Выжившие
columns_train=['Pclass', 'Age'] #Показатели
x = train[columns_train]
y = train[columns_target]
y=y.squeeze()
#разделяем датасет на тренировочную (70%) и тестовую (30%) выборки с помощью
#метода train_test_split из sklearn.model_selection
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


#создаем модель, импортированной из sklearn. linear_model
log_regression = LogisticRegression()

# обучаем модель на тестовых данных
log_regression.fit(x_train, y_train)

#делаем предсказания
y_pred = log_regression.predict(x_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print("Точность:", metrics.accuracy_score(y_test, y_pred))

x_train, y_train = x, y
x_test=test[columns_train]
log_regression.fit(x_train, y_train)
y_pred = log_regression.predict(x_test)
y_predskaz = log_regression.predict(x_train)
print(metrics.confusion_matrix(y_train, y_predskaz))
print("Точность:", metrics.accuracy_score(y_train, y_predskaz))


#создадим людей
df = pd.DataFrame(columns=['Pclass', 'Age'])
#создадим возрастного человека из 3 класса
new_row = {'Pclass': 3, 'Age': 99}
df.loc[1]=new_row
print(df)
print("Данный человек скорее всего... Результат:", Life(log_regression.predict(df)))


# Попробуем взять другие показатели

columns_target=["Survived"] #Выжившие
columns_train=['Sex', 'Age', 'Fare'] #Показатели
train['Sex'] = train['Sex'].replace({'male': 0, 'female': 1})
x = train[columns_train]
y = train[columns_target]
y=y.squeeze()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

log_regression = LogisticRegression()

# обучаем модель на тестовых данных
log_regression.fit(x_train, y_train)

#делаем предсказания
y_pred = log_regression.predict(x_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Матрица ошибок:", cnf_matrix, sep="\n")
print("Точность:", metrics.accuracy_score(y_test, y_pred))

#создадим людей
df = pd.DataFrame(columns=['Sex', 'Age', 'Fare'])
#создадим женщину
new_row = {'Sex': 0, 'Age': 19, 'Fare': 100}
df.loc[1]=new_row
print(df)
print("Данный человек скорее всего... Результат:", Life(log_regression.predict(df)))