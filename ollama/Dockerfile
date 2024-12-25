FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y curl && \
    curl -L https://ollama.ai/install.sh | bash

# On expose le port 11411
EXPOSE 11411

# On définit la variable OLLAMA_HOST pour qu'il écoute sur 0.0.0.0:11411
ENV OLLAMA_HOST=0.0.0.0:11411

# Ensuite, on lance ollama serve
ENTRYPOINT ["/usr/local/bin/ollama", "serve"]
