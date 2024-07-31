# def sell(MY_TREE, sum):

s = input()
length = len(s)
integers = []
i = 0  # индекс текущего символа
while i < length:
    s_int = ''  # строка для нового числа
    while i < length and '0' <= s[i] <= '9':
        s_int += s[i]
        i += 1
    i += 1
    if s_int != '':
        integers.append(int(s_int))
n, k = integers[0], integers[1]


length = k
ak = []
i = 0  # индекс текущего символа
while i < length:
    ak.append(str(input()))
    i+=1

length = n
MY_TREE = []
parents = set()
i = 0  # индекс текущего символа
while i < length:
    integers = []
    TREE = str(input())
    num=1
    for j, element in enumerate(TREE):
        if TREE[j]!=' ' and num!=3:
            integers.append(int(TREE[j]))
            num+=1
        elif TREE[j]!=' ':
            integers.append(str(TREE[j]))
    MY_TREE.append(integers)
    parents.add(integers[0])
    i+=1
money = -1
word = ak
for h in range(n):
    MY_TREE[n-h][0]

#             if word.count(j)>0:
#                 word.remove(j)

print(MY_TREE)
print(parents)
print(MY_TREE[0])