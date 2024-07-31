# Сортировка Вставкой
# def ins_sort(aray):
#     len_list = len(aray)
#     for i in range(1, len_list):
#         ref_value = aray[i]
#         x = i
#         while x>0 and aray[x-1] > ref_value:
#             aray[x] = aray[x-1]
#             x -= 1
#         else:
#             aray[x] = ref_value
#     return aray
#
# aray=[6,5,3,1,2]
# print("before sort ",aray)
# hello=ins_sort(aray)
# print("result sorted is ",hello)

n = int(input())
a = input().split()

for i in range(1, n):
    x = int(a[i])
    j = i

    while j > 0 and int(a[j - 1]) > x:
        a[j] = int(a[j - 1])
        j -= 1
    else:
        a[j] = x

for i in a:
    print(i, end=' ')