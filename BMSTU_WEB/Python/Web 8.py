# С клавиатуры вводится дата в формате DD-MM-YYYY.
# Нужно вывести дату начала недели, к которой относится введенная дата
# (дата понедельника недели), в таком же формате.

from datetime import datetime, timedelta

s = input()
date = datetime.strptime(s, "%d-%m-%Y")
date = (date - timedelta(datetime.weekday(date)))
print(date.strftime("%d-%m-%Y"))
# print(f"{date.day}-{date.month}-{date.year}")