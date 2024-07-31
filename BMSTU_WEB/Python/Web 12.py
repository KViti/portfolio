# Написать функцию, которая на вход принимает строку, а на выход выдает булево значение (True или False),
# которое истинно, если полученная строка соответствует российскому номеру телефона или адресу электронной почты.
# Сигнатура функции:
# check_string(string) -> bool

import re
import numpy as np

# def check_string(string) -> bool:
#     pattern_list = []
#     # номера телефонов
#     # ams='[ ]'
#     pattern_list.append(re.compile(r'^([+]?7|8)?(\d{3}|[(]\d{3}[])])(\d{7})$'))  # без всего лишнего, возможен код оператора в скобках
#     pattern_list.append(re.compile(r'^([+]?7|8)?[(]\d{3}[)]\d{3}-\d{2}-\d{2}$'))  # без пробелов и две тире в конце
#     pattern_list.append(re.compile(r'^([+]?7[ ]|8[ ])?[(]\d{3}[)][ ]\d{3}-\d{2}-\d{2}$'))  # с пробелами и две тире в конце
#     pattern_list.append(re.compile(r'^([+]?7-|8-)?(\d{3}|[(]\d{3}[)])-\d{3}-\d{2}-\d{2}$'))  # все тире
#
#     # почта
#     pattern_list.append(re.compile(r'^\w+([.]\w{2,})*@\w+([.]\w{2,})+$'))
#
#     for pattern in pattern_list:
#         if (pattern.match(string)):
#             return True
#     return False


def check_string(string):
    pattern_list = []
    ams='[ ]'
    # номера телефонов
    pattern_list.append(re.compile(r'^([+]?7|8)?(\d{3}|[(]\d{3}[])])(\d{7})$'))  # без всего лишнего, возможен код оператора в скобках
    pattern_list.append(re.compile(r'^([+]?7|8)?[(]\d{3}[)]\d{3}-\d{2}-\d{2}$'))  # без пробелов и две тире в конце
    pattern_list.append(re.compile(r'^([+]?7ams|8[ ])?[(]\d{3}[)][ ]\d{3}-\d{2}-\d{2}$'))  # с пробелами и две тире в конце
    pattern_list.append(re.compile(r'^([+]?7-|8-)?(\d{3}|[(]\d{3}[)])-\d{3}-\d{2}-\d{2}$'))  # все тире

    # почта
    pattern_list.append(re.compile(r'^\w+([.]\w{2,})*@\w+([.]\w{2,})+$'))

    for pattern in pattern_list:
        if (pattern.match(string)):
            return True
    return False


listok=['89160000000', '+79160000000', '9160000000', '8(916)000-00-00',
        '+7(916)000-00-00', '(916)000-00-00', '8 (916) 000-00-00',
        '+7 (916) 000-00-00', '(916) 000-00-00', '8(916)0000000',
        '+7(916)0000000', '(916)0000000', '8-916-000-00-00',
        '+7-916-000-00-00', '916-000-00-00', 'abc@abc.ab',
        'abc@abc.ab.ab', 'a@ab.ab', 'abc.abc@abc.abc',
        # Невалидные адреса:
        '@abc.abc', 'abc@abc', 'abc@abc.a', 'abc@abc.abc.a',
        'abc@abc.', 'abc@abc@abc']
for i in listok:
    print(i, ':', check_string(i))