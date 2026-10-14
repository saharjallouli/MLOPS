"""
deploiement_prefect.py
----------------------
Enregistre les flows comme "deployments" Prefect et les sert (worker).
Le flow `all` est planifié automatiquement tous les jours à 02h00.

Usage (le serveur Prefect doit déjà tourner) :
    prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
    python deploiement_prefect.py
"""

from prefect import serve

from pipeline_prefect import (
    all_flow,
    clean_flow,
    code_flow,
    evaluate_flow,
    install_flow,
    train_flow,
)

if __name__ == "__main__":
    deployments = [
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",  # tous les jours à 02:00
            tags=["full-pipeline", "mlops"],
        ),
        train_flow.to_deployment(name="ml-pipeline-train", tags=["training", "mlops"]),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]
        ),
        code_flow.to_deployment(name="ml-pipeline-code", tags=["quality", "mlops"]),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["setup", "mlops"]),
        clean_flow.to_deployment(
            name="ml-pipeline-clean", tags=["maintenance", "mlops"]
        ),
    ]
    serve(*deployments)
