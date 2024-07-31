# С клавиатуры вводится строка, а затем - подстрока.
# В строке найти все слова, в которых содержится заданная подстрока, и вывести эти слова целиком.
# Если слова повторяются, вывести все повторения.
string = input()
str = input()

word_list = string.split()

for i, word in enumerate(word_list):
    if(str.lower() in word.lower()):
        print(word)

