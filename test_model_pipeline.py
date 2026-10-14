
from model_pipeline import prepare_data, train_model, evaluate_model

DATA_PATH = "Churn_Modelling.csv"


def test_prepare_data():
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_train_model():
    X_train, _, y_train, _ = prepare_data(DATA_PATH)
    model = train_model(X_train, y_train)

    predictions = model.predict(X_train.head(5))

    assert len(predictions) == 5


def test_evaluate_model():
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1
