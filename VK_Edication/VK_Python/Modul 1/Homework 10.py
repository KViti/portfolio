# x='1,000,000'
# print(x[2:9])
# print(x.count('1'))
# print(x * 3)
# age=12
# print('ht,tyjr' if age<18  else 'Dphjcksq')

a = input().lower().split()
d = dict.fromkeys(a)

for i in a:
    if d[i] is None:
        d[i]=1
    else:
        d[i]+=1

sorted_d = sorted(d.items(), key=lambda item: item[1], reverse=True)

for key, val in d.items():
    if val==max(d.values()):
        max_word=key

key_d = [key for key, value in d.items() if value == max(d.values())]

print(max_word, max(d.values()))
print(sorted_d[0][0], max(d.values()))
print(*key_d, max(d.values()))

