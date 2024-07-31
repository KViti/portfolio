import re
b=input().lower()
a = re.split('"!,.?;:#$%^&*()\s"', b)
d = dict.fromkeys(a)
print(d)
# for i in a:
#     if d[i] is None:
#         d[i]=1
#     else:
#         d[i]+=1
#
# for key, val in d.items():
#     if val==max(d.values()):
#         max_word=key
#
# print(max_word, max(d.values()))