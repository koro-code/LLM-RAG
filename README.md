# LLM-RAG

Industrialisation du déploiement d’un LLM en local via des containers. Voici les attentes :

1. Containers à mettre en place :
   o Interface Chainlit + LangChain : Permettre l'interaction avec le modèle via une interface user-friendly.
   o Ollama + Llama (dernière version) : Container optimisé pour exécuter et tester ces modèles.
   o Base de données relationnelle (Postgres) : Pour la gestion structurée des données textuelles et autres métadonnées.
   o Base de données vectorielle : Nécessaire pour stocker et requêter les embeddings générés.
2. Cas d’usage attendu :
   Déployer une solution RAG (Retrieval-Augmented Generation) :
   o Capacité à uploader des fichiers.
   o Possibilité de les interroger directement à travers le pipeline

Bonus :
Trouve moi un cas d'usage pour faire intéragir l'IA et la maison connecté au bureau.

## Structure du projet

```
mon-projet-llm-rag/
├── chainlit/
│ ├── Dockerfile
│ ├── requirements.txt
│ ├── app.py # Fichier principal de l'appli Chainlit/LangChain
│ └── custom_ollama_llm.py
├── ollama/
│ └── models
│     └── blobs
│     └── manifests
│ └── Dockerfile
├── docker-compose.yml
└── README.md
```
