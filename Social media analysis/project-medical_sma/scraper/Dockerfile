FROM sma-base:1.0.0

WORKDIR /

# dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# twint
RUN pip install --upgrade -e git+https://github.com/twintproject/twint@origin/master#egg=twint

EXPOSE 8080

WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN useradd -ms /bin/bash celery
USER celery
