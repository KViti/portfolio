# В школе прошел экзамен по математике. Несколько человек списали решения и были замечены. Этим школьникам поставил 0 баллов. Задача: есть массив с оценками, среди которых есть 0. Необходимо все оценки, равные нулю перенести в конец массива, чтобы все такие школьники оказались в конце списка.
n = int(input())
a = input().split()

k=0
for i in range(n-k):
    for j in range(i+1, n):
        while a[j]==0:
            j+=1
        if int(a[i])==0 and int(a[j])!=0:
            a[j], a[i] = a[i], a[j]
            k+=1
    # for i in a:
    #     print(i, end=' ')
#   print()


for i in a:
    print(i, end=' ')