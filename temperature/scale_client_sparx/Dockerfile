FROM joelee0515/gpio-base:latest
COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["python", "-m", "scale_client", "--config", "./scale_client/config/mqtt_config.yml", "--log-level", "info"]

