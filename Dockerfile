FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY virtualpi.py .
RUN apt-get update -y
RUN apt-get install default-jdk -y
CMD ["python","-u","virtualpi.py","./pdfs"]
