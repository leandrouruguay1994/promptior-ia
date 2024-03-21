FROM python:3.11.3
ENV PYTHONUNBUFFERED True

RUN pip install --upgrade pip
RUN pip install pydantic==1.10.13
COPY requirements.txt .
RUN pip install --no-cache-dir -r  requirements.txt

ENV APP_HOME /my-app
WORKDIR $APP_HOME
COPY /app $APP_HOME/app

COPY app ./app
COPY packages ./packages
COPY data.txt ./
COPY .env ./

EXPOSE 8080
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]