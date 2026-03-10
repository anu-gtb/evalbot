FROM python:3.12-alpine

WORKDIR /scoring

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY model.py /scoring/
COPY scoring.py /scoring/

EXPOSE 8501

CMD ["streamlit","run","scoring.py","--server.port","8501"]