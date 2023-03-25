''' 
The trained VGG16 model is capable of accurately classifying images of cats and dogs, while the Telegram bot allows for seamless integration and communication with users. 
The implementation is made efficient through the use of PyTorch, a powerful deep learning framework, and the Telebot library, a user-friendly Telegram bot API.
'''
import os
import urllib
import telebot
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO

token = 'TELEGRAM_BOT_TOKEN'
# Загрузка модели VGG16
model = torch.hub.load('pytorch/vision', 'vgg16', pretrained=True)
model.eval()



# Функция для классификации изображения
def classify_image(image):
    # Преобразование изображения в тензор PyTorch и нормализация
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = transform(image).unsqueeze(0)

    # Классификация изображения с помощью модели
    with torch.no_grad():
        output = model(image)

    # Получение метки класса и вероятности
    _, predicted = torch.max(output.data, 1)
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    probability = probabilities[predicted[0]].item()

    # Возврат результата
    print(predicted.item())
    if predicted.item() < 279:
        print("Это собака с вероятностью {:.2f}%".format(probability * 100))
        print('')
        return "Это собака с вероятностью {:.2f}%".format(probability * 100)
    else:
        print("Это кошка с вероятностью {:.2f}%".format((1 - probability) * 100))
        print('')
        return "Это кошка с вероятностью {:.2f}%".format((1 - probability) * 100)



# Создание бота
bot = telebot.TeleBot(token)

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я бот, который может отличать собак от кошек. Пришли мне фото своего питомца и я скажу, кто на нем изображен.\n\nУчти, что бот реагирует только на изображения.\n\nСамые смешные реакции бота (даже не на животных) кидай мне в личку :)")

# Обработчик изображения
@bot.message_handler(content_types=['photo'])
def handle_photo(message):

    # Загрузка изображения из сообщения
    file_info = bot.get_file(message.photo[-1].file_id)
    image_url = 'https://api.telegram.org/file/bot{0}/{1}'.format(token, file_info.file_path)
    with urllib.request.urlopen(image_url) as url:
        image_file = BytesIO(url.read())
    image = Image.open(image_file)

    # Классификация изображения и отправка результата
    result = classify_image(image)
    bot.send_message(message.chat.id, result)

# Запуск бота
bot.polling()

