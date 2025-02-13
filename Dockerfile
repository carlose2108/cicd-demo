FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y python3.9 python3-pip

WORKDIR /app

COPY app/ /app/app
COPY data/ /app/data
COPY run_app.py /app
COPY requirements.txt /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8080
ENTRYPOINT ["python3", "/app/run_app.py"]
