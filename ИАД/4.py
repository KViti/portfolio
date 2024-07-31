import pandas as pd
from sklearn import metrics
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
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

train['Sex'] = train['Sex'].map({'male': 0, 'female': 1}).astype(int)
columns_target=["Survived"] #Выжившие
x = train.loc[:, ~train.columns.isin(['Survived', 'Cabin', 'Embarked', 'Name', 'Ticket'])]
y = train[columns_target]
y = y.values.ravel()
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
#разделяем датасет на тренировочную (70%) и тестовую (30%) выборки с помощью
#метода train_test_split из sklearn.model_selection
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=0)
# Проверим значения k=1...21
k_values = list(range(1, 21))

# Словарь для хранения результатов кросс-валидации для каждого значения
cv_scores = {}

# Цикл для подбора оптимального значения k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_scaled, y, cv=5, scoring='accuracy')
    cv_scores[k] = scores.mean()
# Выбор оптимального значения к, которое дает наилучшую производительность
best_k = max(cv_scores, key=cv_scores.get)
print("Оптимальное значение k:", best_k)
print("Средняя точность при оптимальном k:", cv_scores[best_k])
classifier = KNeighborsClassifier(n_neighbors=best_k)
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)
print(y_test)
print(predictions)

#матрица неточностей
cnf_matrix = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(cnf_matrix, annot=True, fmt="d", cmap="Purples")
plt.xlabel('Predicted values')
plt.ylabel('True values')
plt.title("Error matrix")
plt.show()

print(f"Точность предсказания:{round(metrics.accuracy_score (y_test, predictions),4)}")

y_pred_prob = classifier.predict_proba(x_test)
print (f"ROC AUC:{round(metrics.roc_auc_score(y_test,y_pred_prob[:,1]),4)}")
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob[:,1])
roc_auc = metrics.auc(fpr,tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color = 'darkorange', label ='ROC curve(area- 80.2f)' %roc_auc)
plt.plot([0, 1], [0, 1], '--', color = 'navy')
plt.xlim([0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# Уменьшение размерности данных до двух измерений с помощью РСА pca = PCA(n components=2)

def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(x_scaled)
classifier.fit(X_pca,y)
xx, yy = get_grid(X_pca)
predicted = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, predicted, cmap='autumn')
plt.scatter(X_pca[:, 0],X_pca[:, 1], c=y, s=100, cmap='autumn', edgecolors='black', linewidth=1.5);
plt.xlabel('Principal Component l')
plt.ylabel('Principal Component 2')
plt.title('Decision boundary and data points')
plt.show()