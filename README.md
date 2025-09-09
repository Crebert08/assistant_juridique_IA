# Assistant Code Pénal Burundais

Ce projet est une application web qui agit comme un assistant juridique spécialisé dans le Code pénal du Burundi. Il utilise un modèle de langage avancé (LLM) et une technique de Retrieval-Augmented Generation (RAG) pour répondre aux questions des utilisateurs en se basant sur le texte du Code pénal.

## Comment ça marche ?

L'application est composée de deux parties principales : un backend en Python (FastAPI) et un frontend en JavaScript (React).

### Backend

Le backend gère la logique de traitement du langage naturel. Voici les étapes clés de son fonctionnement :

1.  **Chargement du document** : Au démarrage, le serveur charge le document `Burundi_Code_2017_penal.pdf`.
2.  **Découpage du texte** : Le texte du PDF est découpé en petits segments (chunks) pour être plus facilement traitable.
3.  **Création des embeddings** : Chaque segment de texte est transformé en un vecteur numérique (embedding) à l'aide du modèle `models/embedding-001` de Google.
4.  **Stockage vectoriel** : Ces embeddings sont stockés dans une base de données vectorielle (ChromaDB), qui permet de retrouver rapidement les segments de texte les plus pertinents pour une question donnée.
5.  **API** : Le backend expose une API avec un endpoint `/ask` qui attend une question de l'utilisateur.

Lorsque l'API reçoit une question, le système recherche les segments de texte les plus pertinents dans la base de données vectorielle et les utilise comme contexte pour que le modèle de langage `gemini-pro` génère une réponse précise et contextuelle.

### Frontend

Le frontend est une interface de chat simple et intuitive construite avec React. Elle permet à l'utilisateur de poser des questions et d'afficher les réponses de l'assistant. L'application envoie les questions de l'utilisateur à l'endpoint `/ask` du backend et affiche la réponse reçue.

## Comment démarrer l'application

Suivez ces étapes pour lancer l'application sur votre machine locale.

### Prérequis

- Python 3.7+
- Node.js et npm
- Un compte Google Cloud avec l'API Generative Language activée.

### 1. Configuration de l'environnement

Tout d'abord, clonez le projet sur votre machine.

Créez un fichier `.env` à la racine du projet et ajoutez votre clé d'API Google :

```
GOOGLE_API_KEY="VOTRE_CLE_API_GOOGLE"
```

### 2. Démarrer le Backend

1.  **Ouvrez un terminal** à la racine du projet.
2.  **Créez et activez un environnement virtuel** (recommandé) :
    ```bash
    python -m venv venv
    # Sur Windows
    venv\Scripts\activate
    # Sur macOS/Linux
    source venv/bin/activate
    ```
3.  **Installez les dépendances Python** :
    ```bash
    pip install -r requirements.txt
    ```
4.  **Lancez le serveur FastAPI** :
    ```bash
    uvicorn app.main:app --reload
    ```
Le serveur backend sera accessible à l'adresse `http://localhost:8000`.

### 3. Démarrer le Frontend

1.  **Ouvrez un deuxième terminal**.
2.  **Naviguez vers le dossier `frontend`** :
    ```bash
    cd frontend
    ```
3.  **Installez les dépendances Node.js** :
    ```bash
    npm install
    ```
4.  **Lancez l'application React** :
    ```bash
    npm start
    ```
L'application web s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:3000`.

## Technologies utilisées

- **Backend** :
  - FastAPI
  - LangChain
  - Google Generative AI
  - ChromaDB
  - PyPDFLoader

- **Frontend** :
  - React
  - Axios
