n = int(input())
maxim = []
for i in range(n):
    maximum=1
    s=input()
    for k in s.split(" "):
        if int(k)>maximum:
            maximum=int(k)
    maxim.insert(i, maximum)
#
# while True:
#     s=input()
#     if not s:
#          break
maxim.sort(reverse=True)
print(*maxim, sep=";")
