# подключаем библиотеку машинного зрения
import cv2
# библиотека для вызова системных функций
import os
import json

# получаем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))
# указываем, что мы будем искать лица по примитивам Хаара
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# счётчик изображений
i=0
# расстояния от распознанного лица до рамки
offset=50
# запрашиваем номер пользователя
# name=input('Введите номер пользователя: ')
# получаем доступ к камере
video=cv2.VideoCapture(0)

# Загружаем словарь из файла, если он существует
try:
    with open('user_data.json', 'r') as f:
        user_data = json.load(f)
        # Преобразуем ключи в целые числа для удобства работы
        user_data = {int(k): v for k, v in user_data.items()}
except FileNotFoundError:
    user_data = {}

# Определяем новый ID пользователя автоматически
if user_data:
    user_id = max(user_data.keys()) + 1  # Генерируем следующий ID
else:
    user_id = int(1)  # Начинаем с 1, если данных нет

# Запрашиваем имя пользователя
name = input(f"Введите имя для пользователя {user_id}: ")

# Проверяем, существует ли уже этот ID (хотя это не должно произойти)
if user_id in user_data:
    print("ID уже существует. Выберите другой ID.")
else:
    # Добавляем нового пользователя в словарь
    user_data[user_id] = name
    print(f"Данные для пользователя {user_id} сохранены.")

    # Сохраняем словарь в файл сразу после добавления данных пользователя
    with open('user_data.json', 'w') as f:
        json.dump(user_data, f, indent=4)

    
# запускаем цикл
while True:
    # берём видеопоток
    ret, im =video.read()
    if not ret:
        print("Не удалось получить изображение с камеры.")
        break
    # переводим всё в ч/б для простоты
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # настраиваем параметры распознавания и получаем лицо с камеры
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
    # обрабатываем лица
    for(x,y,w,h) in faces:
        # увеличиваем счётчик кадров
        i=i+1
        # записываем файл на диск
        cv2.imwrite("dataSet/face-"+str(user_id)+'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
        # формируем размеры окна для вывода лица
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        # показываем очередной кадр, который мы запомнили
        cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])
        # делаем паузу
        cv2.waitKey(100)
    # если у нас хватает кадров
    if i>50:
        # освобождаем камеру
        video.release()
        # удалаяем все созданные окна
        cv2.destroyAllWindows()
        # останавливаем цикл
        break
