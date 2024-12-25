# подключаем библиотеку компьютерного зрения
import cv2
# библиотека для вызова системных функций
import os
import json

# получаем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))
# создаём новый распознаватель лиц
recognizer = cv2.face.LBPHFaceRecognizer_create()
# добавляем в него модель, которую мы обучили на прошлых этапах
recognizer.read(path+r'/trainer.yml')
# указываем, что мы будем искать лица по примитивам Хаара
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# получаем доступ к камере
cam = cv2.VideoCapture(0)
# настраиваем шрифт для вывода подписей
font = cv2.FONT_HERSHEY_SIMPLEX


# Загружаем словарь из файла user_data.json
try:
    with open('user_data.json', 'r') as f:
        user_names = json.load(f)
except FileNotFoundError:
    print("Файл user_data.json не найден. Запустите face_gen.py для создания данных.")
    exit()
    
# def capture_and_recognize_faces():
#     cam = cv2.VideoCapture(0)
#     ret, im = cam.read()
#     if not ret:
#         print("Error: Could not read frame from camera.")
#         return None

#     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

#     for (x, y, w, h) in faces:
#         nbr_predicted, coord = recognizer.predict(gray[y:y+h, x:x+w])
#         max_confidence = 150
#         confidence_percentage = 100 * (1 - (coord / max_confidence))
#         threshold = 50
        
#         if confidence_percentage < threshold:
#             name = "Unknown"
#         else:
#             str_id = str(nbr_predicted)
#             name = user_names.get(str_id, "Unknown")

#         cv2.rectangle(im, (x-10, y-10), (x+w+10, y+h+10), (225, 0, 0), 2)
#         cv2.putText(im, str(name), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

#     # Сохраняем изображение с распознанными лицами
#     result_image_path = "recognized_faces.jpg"
#     cv2.imwrite(result_image_path, im)

#     cam.release()
#     return result_image_path
  
# запускаем цикл
while True:
	# получаем видеопоток
	ret, im =cam.read()
	if not ret:
		print("Error: Could not read frame from camera.")
		break
	# переводим его в ч/б
	gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	# определяем лица на видео
	faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
	# перебираем все найденные лица
	for(x,y,w,h) in faces:
		# получаем id пользователя
		nbr_predicted,coord = recognizer.predict(gray[y:y+h,x:x+w])
		# рисуем прямоугольник вокруг лица
		cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
		# если мы знаем id пользователя
		# Нормализация confidence
		max_confidence = 150  # Это значение нужно подбирать экспериментально
		confidence_percentage = 100 * (1 - (coord / max_confidence))

		# Установка порога уверенности
		threshold = 50  # Порог для определения "неизвестного"
		
		if confidence_percentage < threshold:
			name = "Unknown"
		else:
			str_id = str(nbr_predicted)
			if str_id in user_names:
				name = user_names[str_id]
			else:
				name = "Unknown"
			
		#if(nbr_predicted==1):
			 # подставляем вместо него имя человека
			# nbr_predicted='Gekk Sofya'
		# добавляем текст к рамке
		cv2.putText(im,str(name), (x,y+h),font, 1.1, (0,255,0))
		# выводим окно с изображением с камеры
		cv2.imshow('Face recognition',im)
		cv2.waitKey(1)

# Освобождаем ресурсы
cam.release()
cv2.destroyAllWindows()
