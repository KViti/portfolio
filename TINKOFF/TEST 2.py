

n=int(input())
otvet = []
for j in range(n):
    t = int(input())
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

    if sum(integers)>=(2*t-2):
        otvet.append('YES')
    else:
        otvet.append('NO')
for i in otvet:
    print(i)