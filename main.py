"""
main.py
-------
Point d'entrée CLI du pipeline.

Exemples :
    python main.py --train                 # prépare, entraîne, évalue, sauvegarde
    python main.py --evaluate              # charge le modèle et l'évalue
    python main.py --predict               # prédiction sur un exemple
    python main.py --train --evaluate      # combinaisons possibles
"""

import argparse

import pandas as pd

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

# Client exemple du notebook (même ordre que les colonnes de X)
EXAMPLE = {
    "CreditScore": 850, "Gender": 0, "Age": 43, "Tenure": 2,
    "Balance": 125510.82, "NumOfProducts": 1, "HasCrCard": 1,
    "IsActiveMember": 1, "EstimatedSalary": 79084.10,
}


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline ML - Churn Modelling")
    p.add_argument("--data", default="Churn_Modelling.csv", help="Chemin du CSV")
    p.add_argument("--model", default="classifier.joblib", help="Chemin du modèle")
    p.add_argument("--train", action="store_true", help="Entraîner et sauvegarder")
    p.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    p.add_argument("--predict", action="store_true", help="Prédire sur un exemple")
    return p.parse_args()


def main():
    args = parse_args()
    if not (args.train or args.evaluate or args.predict):
        print("Aucune action choisie. Utilisez --train, --evaluate ou --predict.")
        return

    X_train, X_test, y_train, y_test = prepare_data(args.data)
    print(f"Données prêtes : {len(X_train)} train / {len(X_test)} test")

    if args.train:
        model = train_model(X_train, y_train)
        save_model(model, args.model)
        evaluate_model(model, X_test, y_test)
    else:
        model = load_model(args.model)
        if args.evaluate:
            evaluate_model(model, X_test, y_test)

    if args.predict:
        pred = model.predict(pd.DataFrame([EXAMPLE]))[0]
        print("Prédiction (1 = quitte la banque, 0 = reste) :", pred)


if __name__ == "__main__":
    main()
