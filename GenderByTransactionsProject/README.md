Итоговый проект курса "Машинное обучение в бизнесе"

Стек:

ML: sklearn, pandas, numpy, catboost
API: flask  
Данные: с [kaggle](https://www.kaggle.com/c/python-and-analyze-data-final-project/overview)

Задача: Предсказать по пользовательским транцакциям вероятность принадлежности пользователя к полу "1" (Бинарная классификация)

Используемые признаки:

    mcc_code (text)
    tr_type (text)

Преобразования признаков: tfidf

Модель: catboost classifier

Клонировать репозиторий и создать контейнер 

$ docker build -t kate-ds/gender-tr:v1 .

Запускаем контейнер

$ docker run -d -p 8180:8180 kate-ds/gender-tr:v1

Переходим на localhost:8181