# import time
#
# def time_decorator(f):
#     def wrapper(*args):
#         start_time = time.time()
#         a=f(*args)
#         end_time = time.time()
#         print(format(int(end_time-start_time)))
#         return a
#     return wrapper
#
#
# @time_decorator #Декоратор
# def sleep_1_sec():
#     time.sleep(1)#Симулятор
#     print("function")
#     return 25
#
# result = sleep_1_sec()
# print(result)
from array import *

print("Введите 5 чисел")
a=array('i')
for i in range(5):
    b = int(input())
    a.insert(i, b)
# print(a)
flag=1
k=5
swap=int
while flag:
    flag=0
    for i in range(k - 1):
        if (a[i] < a[i + 1]):
            swap = a[i]
            a[i] = a[i + 1]
            a[i + 1] = swap
            flag = 1
    k-=1

for i in range(5):
    print(a[i])