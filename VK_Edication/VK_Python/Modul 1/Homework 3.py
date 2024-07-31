# var = 'World'
# print('Hello %s' %var)
# print('Hello {}!'.format(var))
# print(f'Hello {var}!')
# L=list()
# print(bool(L))
# Li=list('0')
# print(bool(Li))

a = int(input())
b = int(input())
Ret=True
while True:
    c = input()
    if not c:
        break
    if a>int(c) or int(c)>b:
        Ret=False
print(Ret)