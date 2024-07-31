# С клавиатуры вводятся слова через запятую с пробелом.
# Выведите на экран три наиболее часто встречаемых слова, вместе с количеством этих слов.
# Количество должно быть отделено от слова двоеточием и пробелом. Каждая пара слово-количество должна быть выведена на отдельной строчке.
# Для простоты гарантируется, что в строке нет слов с одинаковой встречаемостью.
import re

string = input()
word_list = re.split(", ", string.lower())
mydict = dict()
for i, word in enumerate(word_list):
    mydict[word] = mydict.get(word, 0) + 1
sorted_mydict = sorted(mydict.items(), key=lambda item: item[1], reverse=True)[:3]
for i, top in enumerate(sorted_mydict):
        print(f"{top[0]}: {top[1]}")
