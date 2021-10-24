FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    python3-setuptools \
    git

COPY ./app /vbs/app
COPY ./requirements.txt /vbs/requirements.txt
COPY ./.env /vbs/.env
COPY ./entrypoint.sh /vbs/entrypoint.sh
COPY ./pipelines /vbs/pipelines/

WORKDIR /vbs

RUN chmod +x entrypoint.sh
RUN python3 -m venv /opt/venv/ && /opt/venv/bin/python -m pip install -r requirements.txt
RUN /opt/venv/bin/python -m pypyr /vbs/pipelines/model-download

CMD ["./entrypoint.sh"]

