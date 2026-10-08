"""
model_pipeline.py
-----------------
Fonctions modulaires du pipeline ML de prédiction du churn client.
Issu de la modularisation du notebook `customer_churn.ipynb`.

Étapes : prepare_data -> train_model -> evaluate_model -> save_model / load_model
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TARGET = "Exited"
COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname", "Geography"]


def prepare_data(csv_path="Churn_Modelling.csv", test_size=0.2, random_state=1):
    """Charge le CSV, nettoie, encode et sépare en train/test.

    Args:
        csv_path (str): chemin du fichier CSV.
        test_size (float): proportion du jeu de test (entre 0 et 1).
        random_state (int): graine pour la reproductibilité.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
        ValueError: si la colonne cible est absente ou si test_size est invalide.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")
    if not 0 < test_size < 1:
        raise ValueError("test_size doit être compris entre 0 et 1 (exclus).")

    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise ValueError(f"Colonne cible '{TARGET}' absente du fichier.")

    # Nettoyage : doublons et valeurs manquantes
    df = df.drop_duplicates()
    df = df.dropna()

    # Encodage de la variable catégorielle Gender (Female/Male -> 0/1)
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

    # Séparation features / cible, suppression des colonnes inutiles
    y = df[TARGET]
    X = df.drop(columns=[TARGET] + [c for c in COLUMNS_TO_DROP if c in df.columns])

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un RandomForestClassifier.

    Args:
        X_train (pd.DataFrame): features d'entraînement.
        y_train (pd.Series): cible d'entraînement.
        n_estimators (int): nombre d'arbres.
        random_state (int): graine aléatoire.

    Returns:
        RandomForestClassifier: modèle entraîné.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train et y_train doivent avoir la même taille.")

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle sur le jeu de test.

    Args:
        model: modèle entraîné (avec méthode predict).
        X_test, y_test: données de test.

    Returns:
        dict: accuracy, matrice de confusion et rapport de classification.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
    }
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print("Matrice de confusion :\n", metrics["confusion_matrix"])
    print("Rapport de classification :\n", metrics["report"])
    return metrics


def save_model(model, path="classifier.joblib"):
    """Sauvegarde le modèle avec joblib.

    Args:
        model: modèle à sauvegarder.
        path (str): chemin du fichier de sortie.
    """
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans : {path}")


def load_model(path="classifier.joblib"):
    """Charge un modèle sauvegardé avec joblib.

    Args:
        path (str): chemin du fichier modèle.

    Returns:
        Le modèle chargé.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path)
