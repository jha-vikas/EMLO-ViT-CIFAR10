FROM python:3.7.10-slim-buster

RUN export DEBIAN_FRONTEND=noninteractive \
    && echo "LC_ALL=en_US.UTF-8" >> /etc/environment \
    && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
    && echo "LANG=en_US.UTF-8" > /etc/locale.conf \
    && apt update && apt install -y locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu && rm -rf /root/.cache/pip

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8


WORKDIR /app

COPY requirements.txt requirements.txt

COPY . .

RUN mkdir -p /app/data

RUN wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip -O app/data/kagglecatsanddogs.zip

ENV PYTHONPATH="/app"

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install /app

CMD gold_train; gold_eval ckpt_fol='.'
