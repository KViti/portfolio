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
n, m = integers[0], integers[1]

k = input()
length = len(k)
gift = []
i = 0  # индекс текущего символа
while i < length:
    k_int = ''  # строка для нового числа
    while i < length and '0' <= k[i] <= '9':
        k_int += k[i]
        i += 1
    i += 1
    if k_int != '':
        gift.append(int(k_int))

RAZNIT = []
i=0
for Summa in range(m+1):
    OSTATOK=Summa
    for i in range(len(gift)):
        if OSTATOK>=gift[i]:
            OSTATOK -= gift[i]
    RAZNIT.append(OSTATOK)

print(max(RAZNIT))
