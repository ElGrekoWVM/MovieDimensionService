# Basis-Image
FROM python:3.8

# Arbeitsverzeichnis setzen
WORKDIR /app

# Abhängigkeiten kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Anwendungscode kopieren
COPY . .

# Exponiere den Standardport für Flask
EXPOSE 5005

# Flask-App starten
CMD ["python", "MovieDimensionService.py"]