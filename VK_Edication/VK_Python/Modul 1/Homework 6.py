s = input()
k=""
num=0
for i in s:
    if(i in "!%#@"):
        num+=1
    else:
        k=k+i
print(num)
print(k.lower())
