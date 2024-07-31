# На клетчатой плоскости закрашено K клеток.
# Требуется найти минимальный по площади прямоугольник, со сторонами, параллельными линиям сетки, покрывающий все закрашенные клетки.
# Формат ввода
# Во входном файле, на первой строке, находится число K (1 ≤ K ≤ 100). На следующих K строках находятся пары чисел Xi и Yi — координаты закрашенных клеток (|Xi|, |Yi| ≤ 109).
# Формат вывода
# Выведите в выходной файл координаты левого нижнего и правого верхнего углов прямоугольника.
N=int(input())
stroka = input()
X, Y = int(stroka.split(' ')[0]), int(stroka.split(' ')[1])
min_X=max_X=X
min_Y=max_Y=Y
for i in range(N-1):
    stroka=input()
    X_new, Y_new = int(stroka.split(' ')[0]), int(stroka.split(' ')[1])
    min_X=min(min_X, X_new)
    max_X=max(max_X, X_new)
    min_Y=min(min_Y, Y_new)
    max_Y=max(max_Y, Y_new)
print(min_X, min_Y, max_X, max_Y)
