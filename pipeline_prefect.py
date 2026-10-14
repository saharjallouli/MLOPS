"""
pipeline_prefect.py
-------------------
Orchestration du pipeline ML avec Prefect (Atelier 3).

Tasks : install, format, qualité, sécurité, tests, prepare, train, save,
        load, evaluate, predict, (clean, fetch_project)
Flows : install, code, train, evaluate, all, clean

Exemples :
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow code
    python pipeline_prefect.py --flow train
    python pipeline_prefect.py --flow evaluate
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from prefect import flow, get_run_logger, task

from main import EXAMPLE
from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = str(PROJECT_DIR / "Churn_Modelling.csv")
MODEL_PATH = str(PROJECT_DIR / "classifier.joblib")

# On ne scanne QUE ces fichiers (sinon venv/ serait analysé : très lent)
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run_command(cmd):
    """Exécute une commande shell, journalise la sortie et échoue si code != 0."""
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if output:
        logger.info("\n%s", output)
    if result.returncode != 0:
        raise RuntimeError(f"Échec ({result.returncode}) : {' '.join(cmd)}")
    return output


# ======================= TASKS : environnement et code =======================
@task(name="Installer les dépendances")
def install_dependencies():
    """pip install -r requirements.txt (avec le Python du venv courant)."""
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code (black)")
def format_code():
    """Reformate automatiquement les 3 fichiers avec black."""
    run_command([sys.executable, "-m", "black", *CODE_FILES])


@task(name="Qualité du code (flake8)")
def check_quality():
    """Analyse statique : style, imports inutiles, erreurs de syntaxe."""
    run_command(
        [sys.executable, "-m", "flake8", "--max-line-length", "88", *CODE_FILES]
    )


@task(name="Sécurité du code (bandit)")
def check_security():
    """Recherche de failles de sécurité (niveau moyen et plus)."""
    run_command([sys.executable, "-m", "bandit", "-q", "-ll", *CODE_FILES])


@task(name="Tests unitaires (pytest)")
def run_unit_tests():
    """Exécute les tests de test_model_pipeline.py."""
    run_command([sys.executable, "-m", "pytest", "-q", "test_model_pipeline.py"])


# ========================== TASKS : data et modèle ===========================
@task(name="Préparation des données")
def prepare_data_task():
    """Charge, nettoie et sépare les données (appelle prepare_data)."""
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
    get_run_logger().info("Train : %s | Test : %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


@task(name="Entraînement du modèle")
def train_model_task(X_train, y_train):
    """Entraîne le Random Forest (appelle train_model)."""
    return train_model(X_train, y_train)


@task(name="Sauvegarde du modèle")
def save_model_task(model):
    """Sauvegarde le modèle avec joblib (appelle save_model)."""
    save_model(model, MODEL_PATH)


@task(name="Chargement du modèle")
def load_model_task():
    """Charge le modèle sauvegardé (appelle load_model)."""
    return load_model(MODEL_PATH)


@task(name="Évaluation du modèle")
def evaluate_model_task(model, X_test, y_test):
    """Évalue le modèle (appelle evaluate_model)."""
    metrics = evaluate_model(model, X_test, y_test)
    get_run_logger().info("Accuracy : %.2f%%", metrics["accuracy"] * 100)
    return metrics


@task(name="Prédiction")
def predict_task(model):
    """Prédit le churn du client exemple défini dans main.py."""
    pred = int(model.predict(pd.DataFrame([EXAMPLE]))[0])
    get_run_logger().info("Prédiction (1 = part, 0 = reste) : %s", pred)
    return pred


@task(name="Nettoyage")
def clean_task():
    """Supprime le modèle et les caches générés."""
    for name in ["classifier.joblib", "__pycache__", ".pytest_cache"]:
        path = PROJECT_DIR / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    get_run_logger().info("Fichiers générés supprimés.")


@task(name="Récupération du projet (git)")
def fetch_project(repo_url):
    """Étape 06 : git pull si le dossier est un dépôt, sinon git clone."""
    if (PROJECT_DIR / ".git").exists():
        run_command(["git", "pull"])
    else:
        run_command(["git", "clone", repo_url, str(PROJECT_DIR)])


# ================================== FLOWS ====================================
@flow(name="install", log_prints=True)
def install_flow():
    """Installe les dépendances."""
    install_dependencies()


@flow(name="code", log_prints=True)
def code_flow():
    """Formatage -> qualité -> sécurité -> tests unitaires."""
    format_code()
    check_quality()
    check_security()
    run_unit_tests()


@flow(name="train", log_prints=True)
def train_flow():
    """Préparation des données + entraînement (+ sauvegarde du modèle)."""
    X_train, X_test, y_train, y_test = prepare_data_task()
    model = train_model_task(X_train, y_train)
    save_model_task(model)
    return model


@flow(name="evaluate", log_prints=True)
def evaluate_flow():
    """Chargement du modèle sauvegardé + évaluation."""
    _, X_test, _, y_test = prepare_data_task()
    model = load_model_task()
    return evaluate_model_task(model, X_test, y_test)


@flow(name="all", log_prints=True)
def all_flow(repo_url: str | None = None):
    """Pipeline complet : (git) -> install -> code -> data -> modèle -> éval."""
    if repo_url:
        fetch_project(repo_url)
    install_dependencies()
    code_flow()
    X_train, X_test, y_train, y_test = prepare_data_task()
    model = train_model_task(X_train, y_train)
    save_model_task(model)
    evaluate_model_task(model, X_test, y_test)
    predict_task(model)


@flow(name="clean", log_prints=True)
def clean_flow():
    """Supprime le modèle et les caches."""
    clean_task()


FLOWS = {
    "all": all_flow,
    "train": train_flow,
    "entrainement": train_flow,  # alias (nom utilisé dans l'énoncé)
    "evaluate": evaluate_flow,
    "code": code_flow,
    "install": install_flow,
    "clean": clean_flow,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline ML orchestré par Prefect")
    parser.add_argument("--flow", required=True, choices=sorted(FLOWS))
    parser.add_argument("--repo-url", help="URL git (uniquement pour --flow all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.flow == "all" and args.repo_url:
        all_flow(repo_url=args.repo_url)
    else:
        FLOWS[args.flow]()
