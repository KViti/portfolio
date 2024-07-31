import re
def MY_MEAN(a):
    ls = []
    for i in a:
        if i != "":
            ls.append(int(i))
    print(sum(ls)/(len(ls)+1))

while True:
    a = input().split()
    if not a:
        break
    MY_MEAN(a)