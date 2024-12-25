FROM python:3.10-slim

# Installer les dépendances système éventuelles
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Créer un répertoire pour l'app
WORKDIR /app

# Copier les requirements
COPY requirements.txt .

# Installer les paquets Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . .

# Lancer Chainlit par défaut (port 8501 par ex.)
CMD ["chainlit", "run", "app.py", "--port", "8501", "--host", "0.0.0.0"]
