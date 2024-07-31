# С клавиатуры вводится строка.
# В строке заменить пробелы звездочкой.
# Если встречается подряд несколько пробелов, то их следует заменить одним знаком "*",
# пробелы в начале и конце строки удалить.

# string = input()
# flag = 0
# kon = len(string)
# nach=0
# while(string[nach]==" "):
#     nach+=1
#
# while(string[kon-1]==" "):
#     kon-=1
# for char in range(nach, kon):
#     if (string[char]==" ")&(flag == 0):
#         flag = 1
#         print("*", end="")
#     elif(string[char]!=" "):
#         flag = 0
#         print(string[char], end="")

s = input()
s = s.strip()
words = s.split()
result = "*".join(words)
print(result)