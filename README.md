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


## Structure du projet

```
mon-projet-llm-rag/
├── chainlit/
│ ├── Dockerfile
│ ├── requirements.txt
│ ├── memory.py
│ └──app.py
├── ollama/
│ └── models
│     └── blobs
│     └── manifests
│ └── Dockerfile
├── docker-compose.yml
└── README.md
```

Bienvenue dans le projet **LLM-RAG** ! Cette documentation vous guidera à travers l'installation, la configuration et le fonctionnement de l'application.

## Table des Matières

1. [Installation et Initialisation](#installation-et-initialisation)
2. [Changer le Modèle Installé](#changer-le-modèle-installé)
3. [Fonctionnement de l'Application](#fonctionnement-de-lapplication)
4. [Technologies Utilisées](#technologies-utilisées)


## Installation et Initialisation

Suivez les étapes ci-dessous pour installer et initialiser le projet sur votre machine locale.

### Prérequis

- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/)
- [Ollama](https://ollama.com/)
- Une clé API OpenAI valide

### Étapes d'Installation

1. **Cloner le Répertoire**

   ```bash
   git clone git@github.com:koro-code/LLM-RAG.git
   cd LLM-RAG
   ```

2. **Configurer la Clé OpenAI**

   Ouvrez le fichier docker-compose.yaml et ajoutez votre clé OpenAI à l'emplacement spécifié

3. **Construire les Conteneurs Docker et démarrer les Services**

   ```bash
   docker-compose build
   docker-compose up -d
   ```

4. **Accéder à l'Application**

   [Accéder à l'application](http://localhost:8501/)


## Changer le Modèle Installé

Changer le Modèle Installé
Si vous souhaitez utiliser un modèle différent de celui installé par défaut, suivez les étapes suivantes :


   Téléchargez et installez Ollama si ce n'est pas déjà fait.

   Télécharger le Modèle Souhaité

   ```bash
   ollama pull <NOM-DU-MODEL>
   ```

   Déplacez le modèle téléchargé dans le dossier ollama/models/ de votre projet

   Après avoir ajouté le nouveau modèle, reconstruisez et redémarrez les conteneurs Docker :

   ```bash
   docker-compose build
   docker-compose up -d
   ```

## Fonctionnement de l'Application


# Composants Principaux

- Chainlit : Utilisé pour l'interface utilisateur. Il permet de créer des interfaces interactives et réactives pour les utilisateurs finaux.

- LangChain : Intégré pour la gestion et l'orchestration des modèles de langage. LangChain facilite la manipulation des modèles et leur intégration dans le flux de travail de l'application.

- Ollama : Responsable de la récupération et de la gestion des modèles de langage. Ollama simplifie le processus de téléchargement et de mise à jour des modèles.

- ChromaDB : Base de données vectorielle utilisée pour le stockage des fichiers. ChromaDB permet un stockage efficace et rapide des données vectorielles nécessaires au fonctionnement des modèles.

- PostgreSQL : Utilisé pour stocker l'historique des interactions en base de données relationnelle. PostgreSQL assure la persistance et la fiabilité des données historiques de l'application.