import numpy as np

n=int(input())
otvet = []
for i in range(n):
    strochka = str(input())
    word = ['T', 'I', 'N', 'K', 'O', 'F', 'F']
    if len(strochka)==len(word):
        for j in strochka:
            if word.count(j)>0:
                word.remove(j)
    if word:
        otvet.append('No')
    else:
        otvet.append('YES')
for i in otvet:
    print(i)