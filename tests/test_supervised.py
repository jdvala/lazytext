import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from lazytext.supervised import LazyTextPredict


@pytest.fixture()
def data():
    data = pd.read_csv("tests/assets/bbc-text.csv")
    data.dropna(inplace=True)
    return data


def process_data(train_data, test_data):
    vectorizer = TfidfVectorizer(stop_words="english")
    x_train = vectorizer.fit_transform(train_data.text)
    x_test = vectorizer.transform(test_data.text)
    y_train = train_data.category.tolist()
    y_test = test_data.category.tolist()
    return x_train, x_test, y_train, y_test


def test_lazytext(data):
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=13)
    x_train, x_test, y_train, y_test = process_data(train_data, test_data)

    lazy_text = LazyTextPredict(
        classifiers=["MultinomialNB"], classification_type="multiclass"
    )
    models = lazy_text.fit(x_train, x_test, y_train, y_test)

    assert models[0]["name"] == "MultinomialNB"
    assert isinstance(models[0]["model"], type(MultinomialNB()))


def test_lazytext_custom_metric(data):
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=13)
    x_train, x_test, y_train, y_test = process_data(train_data, test_data)

    def my_very_sophisticated_metric(y_true, y_pred):
        score = accuracy_score(y_true=y_true, y_pred=y_pred)
        return score

    lazy_text = LazyTextPredict(
        classifiers=["MultinomialNB"],
        classification_type="multiclass",
        custom_metric=my_very_sophisticated_metric,
    )
    models = lazy_text.fit(x_train, x_test, y_train, y_test)
    assert models[0]["custom_metric_score"] != "NA"


def test_lazytext_custom_parameters(data):
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=13)
    x_train, x_test, y_train, y_test = process_data(train_data, test_data)

    custom_parameters = [
        {"name": "SVC", "parameters": {"C": 0.5, "kernel": "poly", "degree": 5}}
    ]

    lazy_text = LazyTextPredict(
        classifiers=["SVC"],
        classification_type="multiclass",
        custom_parameters=custom_parameters,
    )
    models = lazy_text.fit(x_train, x_test, y_train, y_test)
    assert models[0]["model"].__dict__["C"] == custom_parameters[0]["parameters"]["C"]
    assert (
        models[0]["model"].__dict__["kernel"]
        == custom_parameters[0]["parameters"]["kernel"]
    )
