FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt
COPY packages.txt packages.txt

# we need to install the packages without versions
# to ensure compatibility with apple ARM devices
RUN sed -i 's/==.*//' requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN rm -rf /root/.cache/pip

RUN apt-get update
RUN xargs apt-get -y install < packages.txt

COPY datachad datachad
COPY app.py app.py

ARG port=80
ENV STREAMLIT_SERVER_PORT ${port}
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
EXPOSE ${port}

CMD ["streamlit", "run", "app.py"]