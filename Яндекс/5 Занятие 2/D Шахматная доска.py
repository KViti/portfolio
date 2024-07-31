# Из шахматной доски по границам клеток выпилили связную (не распадающуюся на части) фигуру без дыр.
# Требуется определить ее периметр.
# Формат ввода
# Сначала вводится число N (1 ≤ N ≤ 64) – количество выпиленных клеток.
# В следующих N строках вводятся координаты выпиленных клеток, разделенные пробелом (номер строки и столбца – числа от 1 до 8).
# Каждая выпиленная клетка указывается один раз.
# Формат вывода
# Выведите одно число – периметр выпиленной фигуры (сторона клетки равна единице).

N = int(input())
P=0
array_X = []
array_Y = []
for i in range(N):
    stroka=input()
    X, Y = int(stroka.split()[0]), int(stroka.split()[1])
    P+=4
    if (X in array_X):
        indexes = [i for i, x in enumerate(array_X) if x == X]
        for i in indexes:
            if Y+1==array_Y[i] or Y-1==array_Y[i]:
                P-=2
    if (Y in array_Y):
        indexes = [i for i, x in enumerate(array_Y) if x == Y]
        for i in indexes:
            if X+1==array_X[i] or X-1==array_X[i]:
                P-=2
    array_X.append(X)
    array_Y.append(Y)
print(P)