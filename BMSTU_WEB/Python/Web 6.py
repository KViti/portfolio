# С клавиатуры вводится строка, содержащая произвольное количество слов через запятую и пробел.
# Слова могут повторяться.
# Нужно вывести на экран все слова в алфавитном порядке по одному разу, также через пробел и запятую.\
# import re
# string = input()
# word_list = re.split(", ", string.lower())
# myset = set()
# for i, word in enumerate(word_list):
#         myset.add(word)
# kon = len(myset)-1
# for i, word in enumerate(sorted(myset)):
#     if i!=kon:
#         print(word, end=", ")
#     else:
#         print(word, end="")


l= input().lower().split(",")
print(*sorted(set(l)), sep=", ")