# Постановка задачи
# Вы с друзьями периодически собираетесь на лавке у подъезда выпить чаю с баранками. Каждый раз собирается разное количество человек, которое должно скинуть по n рублей на мероприятие. Также у вас в компании есть общак, в который можно докинуть денег, а можно взять. Все транзакции в общак логируются, т.е. у вас есть информация о каждом переводе денег в/из общака для каждого из друзей. Ваша задача - посчитать, сколько денег должен каждый из собравшихся на мероприятие друзей на момент начала мероприятия. Будем считать, что все деньги проходят через общак, поэтому у каждого друга к началу мероприятия уже есть некоторый "баланс" в общаке.
#
# Формат хранения данных
# Все операции с деньгами хранятся в базе данных и приходят к вам в виде списка словарей вида:
#
#     {"name": "Василий", "amount": 500},
#     {"name": "Петя", "amount": 100},
#     {"name": "Василий", "amount": -300},
# ]
# где name - имя друга (считаем, что всех зовут по-разному), amount - сумма, которая добавлена в общак. Если сумма отрицательная - это значит, что друг взял деньги из общака. Также считаем, что скидываются суммы в рублях без копеек.
#
# Вам нужно описать две функции.
# get_balance(name, transactions) -> int
# функция, которая возвращает текущий баланс друга с именем name, исходя из списка транзакций transactions. Если имя name ни разу не встречается в списке transactions, считаем, что баланс этого друга в общаке равен 0 рублей.
# count_debts(names, amount, transactions) -> dict
# функция, которая принимает список имен присутствующих на мероприятии друзей names, стоимость баранок и чая на человека amount, а также список транзакций в общак transactions. Вернуть эта функция должна словарь вида {"имя_друга": 100}, где 100 - это количество денег, которое он должен скинуть на мероприятие. Если на балансе друга больше денег, чем требуется на мероприятие, то он должен 0 рублей.
# Формат ввода
# transactions = [ {"name": "Василий", "amount": 500}, {"name": "Петя", "amount": 100}, {"name": "Василий", "amount": -300}, ]
#
# get_balance("Василий", transactions)
#
# count_debts(["Василий", "Петя", "Вова"], 150, transactions)
#
# Формат вывода
# get_balance("Василий", transactions) == 200
#
# count_debts(["Василий", "Петя", "Вова"], 150, transactions) == {"Василий": 0, "Петя": 50, "Вова": 150}

def positive(duty) -> int:
    if duty >= 0:
        return duty
    else:
        return 0
def get_balance(name, transactions) -> int:
    s = 0
    for element in transactions:
        for key in element.values():
            if key == name:
                s += element.get("amount")
    return s


def count_debts(names, amount, transactions) -> dict:
    dictionary = dict()
    for name in names:
        for element in transactions:
            for key in element.values():
                if key == name:
                        dictionary.update({name: positive(amount-get_balance(name, transactions))})
        if name not in dictionary:
            dictionary.update({name: amount})
    return (dictionary)


transactions = [{"name": "Василий", "amount": 500}, {"name": "Петя", "amount": 100},
                {"name": "Василий", "amount": -300}, ]
print(get_balance("Василий", transactions))
print(count_debts(["Василий", "Петя", "Вова"], 150, transactions))