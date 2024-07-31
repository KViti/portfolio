import mysql.connector

# Создать соединение с базой данных
cnx = mysql.connector.connect(
  host="dc-webdev.bmstu.ru",
  port=8080,
  user="student_2023",
  password="qwerty123",
  database="hotel"
)

# Напечатать версию сервера MySQL
print(cnx.get_server_version())

# Закрыть соединение
cnx.close()