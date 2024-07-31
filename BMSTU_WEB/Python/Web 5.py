# С клавиатуры подается 5 чисел, разделенных концом строки.
# Нужно вывести их на экран от большего к меньшему, также разделяя их концом строки.

# from array import *
# a=array('i', [])
# for i in range(5):
#     b = int(input())
#     a.insert(i, b)
# # print(a)
# flag=1
# k=5
# swap=int
# while flag:
#     flag=0
#     for i in range(k - 1):
#         if (a[i] < a[i + 1]):
#             swap = a[i]
#             a[i] = a[i + 1]
#             a[i + 1] = swap
#             flag = 1
#     k-=1
#
# for i in range(5):
#     print(a[i])

from array import *
result = []
for i in range(5):
    result.append(int(input()))
for i in sorted(result)[::-1]:
    print(i)