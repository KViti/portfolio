import random

# создаем новый файл и записыаем в него список
file1 = open("file1.txt", 'w')
spisok = ['Гречка', 'Рис', 'Овсянка', 'Макароны', 'Яйцо', 'Яйцо', 'Яйцо', 'Яйцо', 'Яйцо']

for i in range(len(spisok)):
    spisok[i] = spisok[i] + "\n"
    file1.close()

with open('file1.txt', 'a') as file1:
    file1.writelines(spisok)

# читаем список из файла и сортируем его
with open('file1.txt', 'r') as file1:
    spisok = file1.readlines()
    spisok.sort()

# Записываем отсортированный список в файл
with open('file2.txt', 'w') as file2:
    for spisok in spisok:
        file2.write(str(spisok))
    file2.close()

with open('file2.txt', 'r') as file2:
    print('Чтение из файла:', file2.read(), sep='\n')

# создаем пустое множество продуктов
products = set()

# открываем файл с продуктами для чтения
with open("file2.txt", "r") as file:
    for line in file:
        spisok = line.strip()
        products.add(spisok)

# создаем словарь
products_dict = {product: 0 for product in products}
print('Вывод словаря без цен:', products_dict, end='\n\n')
# присваиваем случайные цены
for i in range(len(products_dict)):
    products_dict[list(products_dict.keys())[i]] = random.randint(5, 10) * 10

print('Вывод словаря с ценами:', products_dict, end='\n\n')

# сортируем словарь по цене и выводим продукты
sorted_products = sorted(products_dict.items(), key=lambda x: x[1])
print('Вывод отсортированного словаря по ценам:', sorted_products, end='\n\n')
print("Продукты по возрастанию цены:")
for product, price in sorted_products:
    print(f"{product} - {price}р.")
