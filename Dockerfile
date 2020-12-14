From ubuntu:latest

RUN apt-get update -y 
RUN apt-get install -y python3-pip python3-dev

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm

EXPOSE 9988 

COPY . /app

ENTRYPOINT ["python3"]

CMD ["flask_app_sentiment.py"]
