# Напишите скрипт, который считывает строку с клавиатуры.
# Выведите на экран "привет", если введенная строка совпадает со строкой "привет" или "здравствуйте".
# Если введено что угодно другое, ничего выводить не нужно.

string = input()
if (string == "привет") or (string == "здравствуйте"):
    print('привет')