# Подключаем модуль для Телеграма
import telebot
# Указываем токен
bot = telebot.TeleBot('7329769633:AAEP1saNyDatm51t4WRG32nUdFGpQ3jHS_g')
# Импортируем типы из модуля, чтобы создавать кнопки
import json
from telebot import types
# import face_recognition
import cv2
import os
from face_training import get_images_and_labels  # Импортируем функцию

user_data = {}
# Загружаем словарь из файла, если он существует
try:
    with open('user_data.json', 'r') as f:
        user_data = json.load(f)
        # Преобразуем ключи в целые числа для удобства работы
        user_data = {int(k): v for k, v in user_data.items()}
except FileNotFoundError:
    user_data = {}
  
markup1 = types.ReplyKeyboardMarkup(resize_keyboard=True)
markup1.add(types.KeyboardButton("Добавить пользователя"), types.KeyboardButton("Распознать пользователя"), types.KeyboardButton("Камера"), types.KeyboardButton("Распознать по видео"))

@bot.message_handler(commands=['start', 'help', 'Привет'])
def send_welcome(message):
    # Пишем приветствие
    bot.send_message(message.from_user.id, "Привет! На данный момент я умею следующее:")
    # Готовим кнопки
    keyboard = types.InlineKeyboardMarkup()
    # По очереди готовим текст и обработчик для каждой функции
    key_1= types.InlineKeyboardButton(text='Добавить пользователя', callback_data='add_user')
    # И добавляем кнопку на экран
    keyboard.add(key_1)
    key_2 = types.InlineKeyboardButton(text='Распознать пользователя', callback_data='recog_user')
    keyboard.add(key_2)
    key_3 = types.InlineKeyboardButton(text='Камера', callback_data='camera')
    keyboard.add(key_3)
    key_3 = types.InlineKeyboardButton(text='Распознать по видео', callback_data='video')
    keyboard.add(key_3)

    # Показываем все кнопки сразу и пишем сообщение о выборе
    bot.send_message(message.from_user.id, text='Выбери одну из доступных функций:', reply_markup=keyboard)

# Обработчик нажатий на кнопки
@bot.callback_query_handler(func=lambda call: True)
def callback_worker(call):
    # Если нажали на одну из 3 кнопок — вызываем функции
    if call.data == "add_user": 
        msg = bot.send_message(call.message.chat.id, "Введите имя пользователя:", reply_markup=markup1)
        bot.register_next_step_handler(msg, process_user_name)

    elif call.data == "recog_user":
        # отправить фото и попробовать его распознать из имеющейся базы
        msg = bot.send_message(call.message.chat.id, "Отправьте фотографию для распознавания.", reply_markup=markup1)
        bot.register_next_step_handler(msg, process_recognition_photo)

    elif call.data == "camera":
        # Вызов функции захвата изображения с камеры
        image_path = capture_and_recognize_faces(call.message)
        
        if image_path:  # Если изображение успешно получено
            with open(image_path, "rb") as photo:
                bot.send_photo(call.message.chat.id, photo)
        else:
            bot.send_message(call.message.chat.id, "Не удалось захватить изображение.")

    elif call.data == "video":
        msg = bot.send_message(call.message.chat.id, "Отправьте видео для распознавания.", reply_markup=markup1)
        bot.register_next_step_handler(msg, process_num_video)

    else:
        bot.send_message(call.message.chat.id, "Я тебя не понимаю. Выбери доступную функцию.")



# ///////////////////
@bot.message_handler(func=lambda message: message.text == "Добавить пользователя")
def get_user_name(message):
    msg = bot.reply_to(message, "Введите имя пользователя:", reply_markup=markup1)
    bot.register_next_step_handler(msg, process_user_name)

def process_user_name(message):
    user_name = message.text
    for existing_user_name in user_data.values():
        if existing_user_name == user_name:
            msg = bot.reply_to(message, f"Пользователь {user_name} уже существует. Придумай другое имя", reply_markup=markup1)
            bot.register_next_step_handler(msg, lambda msg:process_user_name(msg))
            return
    msg = bot.reply_to(message, "Сколько фотографий вы хотите добавить?", reply_markup=markup1)
    bot.register_next_step_handler(msg, lambda msg:process_num_photos(user_name, msg))

def process_num_photos(user_name, message):
  try:
    num_photos = int(message.text)
    if num_photos <= 0:
        bot.reply_to(message, "Введите положительное число фотографий.", reply_markup=markup1)
        bot.register_next_step_handler(message, lambda msg:process_num_photos(user_name, msg))
        return
    msg = bot.reply_to(message, "Отправьте фото пользователя:", reply_markup=markup1)
    bot.register_next_step_handler(msg, lambda msg: process_photo(msg, user_name, num_photos, i=1))
  except ValueError:
        bot.reply_to(message, "Неверный формат числа. Пожалуйста, введите число.", reply_markup=markup1)
        bot.register_next_step_handler(message, lambda msg:process_num_photos(user_name, msg))


def process_photo(message, user_name, expected_photos, i):
    if message.content_type == 'photo':
        if i == 1:
            if user_data:
                user_id = max(user_data.keys()) + 1
            else:
                user_id = 1
            
            add_user_to_database(user_id, user_name)
            
        else: user_id = max(user_data.keys())
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        nparr = np.frombuffer(downloaded_file, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) > 1:
            bot.reply_to(message, "На фотографии несколько лиц. Пожалуйста, отправьте фотографию с одним человеком.", reply_markup=markup1)
            bot.register_next_step_handler(message, lambda msg: process_photo(msg, user_name, expected_photos, i))
            return

        if len(faces) == 0:
            bot.reply_to(message, "На фотографии не обнаружено лиц. Отправьте другую фотографию", reply_markup=markup1)
            bot.register_next_step_handler(message, lambda msg: process_photo(msg, user_name, expected_photos, i))
            return

        for (x, y, w, h) in faces:
            cropped_face = gray[y:y + h, x:x + w]
            file_name = f"dataSet/face-{user_id}.{i}.jpg"
            os.makedirs("dataSet", exist_ok=True)
            cv2.imwrite(file_name, cropped_face)

        if i >= expected_photos: #Проверка на количество файлов в папке
            bot.reply_to(message,"Все фотографии успешно добавлены в базу данных!", reply_markup=markup1)
            # получаем список картинок и подписей
            images, labels = get_images_and_labels(path+r'/dataSet')
            # обучаем модель распознавания на наших картинках и учим сопоставлять её лица и подписи к ним
            recognizer.train(images, np.array(labels))
            # сохраняем модель
            recognizer.save(path+r'/trainer.yml')
            # удаляем из памяти все созданные окнаы
            cv2.destroyAllWindows()
        else:
          bot.reply_to(message, f"Ожидается еще {expected_photos - i} фотографий.", reply_markup=markup1)
          i=i+1
          bot.register_next_step_handler(message, lambda msg: process_photo(msg, user_name, expected_photos, i))

    else:
        bot.reply_to(message, "Вы отправили не фотографию. Пожалуйста, отправьте фотографию.", reply_markup=markup1)
        bot.register_next_step_handler(message, lambda msg: process_photo(msg, user_name, expected_photos, i))



def add_user_to_database(user_id, user_name):
    user_data[user_id] = user_name
    # Сохраняем словарь в файл сразу после добавления данных пользователя
    with open('user_data.json', 'w') as f:
        json.dump(user_data, f, indent=4)

# /////////////////

# получаем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))
# создаём новый распознаватель лиц
recognizer = cv2.face.LBPHFaceRecognizer_create()
# добавляем в него модель, которую мы обучили на прошлых этапах
recognizer.read(path+r'/trainer.yml')
# указываем, что мы будем искать лица по примитивам Хаара
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# Загружаем словарь из файла user_data.json
try:
    with open('user_data.json', 'r') as f:
        user_names = json.load(f)
except FileNotFoundError:
    print("Файл user_data.json не найден. Запустите face_gen.py для создания данных.")
    exit()
  
@bot.message_handler(func=lambda message: message.text == "Распознать пользователя")
def recogn_user(message):
    bot.reply_to(message, "Отправьте фотографию для распознавания.", reply_markup=markup1)
    bot.register_next_step_handler(message, process_recognition_photo)

def process_recognition_photo(message):
    if message.content_type == 'photo':
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open('received_photo.jpg', 'wb') as new_file:
            new_file.write(downloaded_file)
        processed_image = recognize_faces(downloaded_file)

        if processed_image:
            bot.send_photo(message.chat.id, processed_image, caption="Фото с распознанными лицами", reply_markup=markup1)
        else:
            bot.send_message(message.chat.id, "Лица не обнаружены или произошла ошибка.", reply_markup=markup1)
    else:
        bot.reply_to(message, "Ошибка: Отправьте фотографию.", reply_markup=markup1)
        bot.register_next_step_handler(message, process_recognition_photo)
    
font = cv2.FONT_HERSHEY_SIMPLEX
import numpy as np
def recognize_faces(image_bytes):
    """Распознает лица на изображении и возвращает обработанное изображение в виде байтового потока."""
    path = os.path.dirname(os.path.abspath(__file__))
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(path + r'/trainer.yml')
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    font = cv2.FONT_HERSHEY_SIMPLEX

    try:
        with open('user_data.json', 'r') as f:
            user_names = json.load(f)
    except FileNotFoundError:
        print("Файл user_data.json не найден.")
        return None

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        nbr_predicted, coord = recognizer.predict(gray[y:y+h, x:x+w])
        max_confidence = 150
        confidence_percentage = 100 * (1 - (coord / max_confidence))
        threshold = 65

        if confidence_percentage > threshold:
            str_id = str(nbr_predicted)
            if str_id in user_names:
                name = user_names[str_id]
            else:name = "Unknown"
                   
        else:name = "Unknown"
                    
        cv2.rectangle(image, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
        cv2.putText(image, str(name), (x, y + h), font, 1.1, (0, 255, 0))

    _, img_encoded = cv2.imencode('.jpg', image)
    return img_encoded.tobytes()


# //////////////////////
@bot.message_handler(func=lambda message: message.text == "Распознать по видео")
def process_num_video(message):
    msg = bot.reply_to(message, "Отправьте видео:", reply_markup=markup1)
    bot.register_next_step_handler(msg, lambda msg: process_recognition_video(msg))

def recognize_faces_video(image_bytes):
    """Распознает лица на изображении и возвращает обработанное изображение в виде байтового потока."""
    path = os.path.dirname(os.path.abspath(__file__))
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(path + r'/trainer.yml')
    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        with open('user_data.json', 'r') as f:
            user_names = json.load(f)
    except FileNotFoundError:
        print("Файл user_data.json не найден.")
        return None

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Could not decode image.")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    for (x, y, w, h) in faces:
        nbr_predicted, coord = recognizer.predict(gray[y : y + h, x : x + w])
        max_confidence = 150
        confidence_percentage = 100 * (1 - (coord / max_confidence))
        threshold = 50
        if confidence_percentage > threshold:
            str_id = str(nbr_predicted)
            if str_id in user_names:
                name = user_names[str_id]
            else:
                name = "Unknown"

        else:
            name = "Unknown"

        cv2.rectangle(
            image, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2
        )
        cv2.putText(image, str(name), (x, y + h), font, 1.1, (0, 255, 0))
    _, img_encoded = cv2.imencode('.jpg', image)
    return img_encoded.tobytes()


def process_recognition_video(message):
    if message.content_type == 'video':
        try:
            file_info = bot.get_file(message.video.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            video_path = 'received_video.mp4'
            with open(video_path, 'wb') as new_file:
                new_file.write(downloaded_file)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                bot.send_message(message.chat.id, "Ошибка: не удалось открыть видеофайл.", reply_markup=markup1)
                return

            output_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                     break

                _, img_encoded = cv2.imencode('.jpg', frame)
                processed_frame_bytes = recognize_faces_video(img_encoded.tobytes())

                if processed_frame_bytes:
                    nparr = np.frombuffer(processed_frame_bytes, np.uint8)
                    processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    output_frames.append(processed_frame)
                else:
                     output_frames.append(frame) #if face recognition fails, just append the frame without changes

            cap.release()
            if output_frames:
                height, width, _ = output_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_video_path = 'processed_video.mp4'
                out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

                for frame in output_frames:
                    out.write(frame)
                out.release()

                with open(output_video_path, 'rb') as video:
                   bot.send_video(
                    message.chat.id,
                    video,
                    caption="Видео с распознанными лицами",
                     reply_markup=markup1
                   )
                os.remove(output_video_path)
            else:
               bot.send_message(
 message.chat.id,
                 "Лица не обнаружены или произошла ошибка.",
                  reply_markup=markup1
               )
            os.remove(video_path)

        except Exception as e:
             bot.send_message(
                message.chat.id,
                  f"Произошла ошибка при обработке видео: {e}",
                    reply_markup=markup1
            )


    else:
        bot.reply_to(message, "Ошибка: Отправьте видео.", reply_markup=markup1)
        bot.register_next_step_handler(message, process_recognition_video)


ALLOWED_USER_IDS = [7329769633] #ID пользователей с разрешенным доступом 
# /////////////////////
@bot.message_handler(func=lambda message: message.text == "Камера")
def recogn_user(message):
    image_path = capture_and_recognize_faces(message)
        
    if image_path:  # Если изображение успешно получено
        with open(image_path, "rb") as photo:
            bot.send_photo(message.chat.id, photo)
    else:
        bot.send_message(message.chat.id, "Не удалось захватить изображение.")

def capture_and_recognize_faces(message):
    if message.from_user.id in ALLOWED_USER_IDS:
        cam = cv2.VideoCapture(0)
        ret, im = cam.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            return None

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            nbr_predicted, coord = recognizer.predict(gray[y:y+h, x:x+w])
            max_confidence = 150
            confidence_percentage = 100 * (1 - (coord / max_confidence))
            threshold = 65
            
            if confidence_percentage > threshold:
                name = "Unknown"
            else:
                str_id = str(nbr_predicted)
                name = user_names.get(str_id, "Unknown")

            cv2.rectangle(im, (x-10, y-10), (x+w+10, y+h+10), (225, 0, 0), 2)
            cv2.putText(im, str(name), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Сохраняем изображение с распознанными лицами
        result_image_path = "recognized_faces.jpg"
        cv2.imwrite(result_image_path, im)

        cam.release()
        return result_image_path  
    else:
           bot.reply_to(message, "У вас нет доступа к этой команде.")

bot.polling()