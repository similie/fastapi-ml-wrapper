FROM python:3.10

EXPOSE 5002

# Important, uvicorn expects app.main to exist in /code
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /code/app
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5002"]
