import inspect
import logging
import time
import typing as tt
import warnings

import numpy as np
import sklearn
from rich.progress import track
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import all_estimators
from sklearn.utils.class_weight import compute_class_weight

from lazytext.create_table import create_table

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

"""
TODO:
* Check out tqdm to display more information, like which estimators is being fit.
* Write docstrings
* Write tests
"""
DO_NOT_APPLY = [
    "CategoricalNB",
    "ClassifierChain",
    "GaussianNB",
    "GaussianProcessClassifier",
    "HistGradientBoostingClassifier",
    "LabelPropagation",
    "LabelSpreading",
    "LinearDiscriminantAnalysis",
    "MultiOutputClassifier",  # This can be an option as well.
    "OneVsOneClassifier",
    "QuadraticDiscriminantAnalysis",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "StackingClassifier",
    "VotingClassifier",
    "RadiusNeighborsClassifier",
    "RidgeClassifierCV",
]


class LazyTextPredict:
    """This module will fit all the classification algorithms available in Scikit-learn for text classification.

    Args:
        random_state: Set the global random seed for reporducible resutls.
        classifiers: List of estimators to use. Default = ["all"] which uses all the estimators available in scikit learn for text classification.
        classification_type: Type of classification being performed. Default = "binary"
        class_weights: When set to True, class weights for classes will be calculated and will be applied to the estimators accepting the class_weight argument.
                       Default = True. The class weights calculation stretegy by default is set to balanced.
        custom_matric: Custom evaluation metric. Default = None
        custom_parameters: Custom parameters for models. Default = None, which means that parameters for all the estimators will be default ones set by Scikit-learn team.

    Examples:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from lazytext.supervised import LazyTextPredict
        import re
        import nltk

        # Load the dataset
        df = pd.read_csv("tests/assets/bbc-text.csv")
        df.dropna(inplace=True)

        # Download models required for text cleaning
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        # split the data into train set and test set
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=13)

        # Tokenize the words
        df_train['clean_text'] = df_train['text'].apply(nltk.word_tokenize)
        df_test['clean_text'] = df_test['text'].apply(nltk.word_tokenize)

        # Remove stop words
        stop_words=set(nltk.corpus.stopwords.words("english"))
        df_train['text_clean'] = df_train['clean_text'].apply(lambda x: [item for item in x if item not in stop_words])
        df_test['text_clean'] = df_test['clean_text'].apply(lambda x: [item for item in x if item not in stop_words])

        # Remove numbers, punctuation and special characters (only keep words)
        regex = '[a-z]+'
        df_train['text_clean'] = df_train['text_clean'].apply(lambda x: [item for item in x if re.match(regex, item)])
        df_test['text_clean'] = df_test['text_clean'].apply(lambda x: [item for item in x if re.match(regex, item)])

        # Lemmatization
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        df_train['text_clean'] = df_train['text_clean'].apply(lambda x: [lem.lemmatize(item, pos='v') for item in x])
        df_test['text_clean'] = df_test['text_clean'].apply(lambda x: [lem.lemmatize(item, pos='v') for item in x])

        # Join the words again to form sentences
        df_train["clean_text"] = df_train.text_clean.apply(lambda x: " ".join(x))
        df_test["clean_text"] = df_test.text_clean.apply(lambda x: " ".join(x))

        # Tfidf vectorization
        vectorizer = TfidfVectorizer()

        x_train = vectorizer.fit_transform(df_train.clean_text)
        x_test = vectorizer.transform(df_test.clean_text)
        y_train = df_train.category.tolist()
        y_test = df_test.category.tolist()

        lazy_text = LazyTextPredict(
            classification_type="multiclass",
            )
        models = lazy_text.fit(x_train, x_test, y_train, y_test)


              Label Analysis
        ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
        ┃ Classes       ┃ Weights            ┃
        ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
        │ business      │ 0.8725490196078431 │
        │ sport         │ 1.1528497409326426 │
        │ politics      │ 1.0671462829736211 │
        │ entertainment │ 0.8708414872798435 │
        │ tech          │ 1.1097256857855362 │
        └───────────────┴────────────────────┘
                                                                    Result Analysis
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ Model                       ┃ Accuracy           ┃ Balanced Accuracy  ┃ F1 Score            ┃ Custom Metric Score ┃ Time Taken           ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ AdaBoostClassifier          │ 0.7260479041916168 │ 0.717737172132769  │ 0.7248335989941609  │ NA                  │ 1.4244091510772705   │
        │ BaggingClassifier           │ 0.8817365269461078 │ 0.8796633962363677 │ 0.8814695332332374  │ NA                  │ 2.422576904296875    │
        │ BernoulliNB                 │ 0.9535928143712575 │ 0.9505929193425733 │ 0.9533647387436917  │ NA                  │ 0.015914201736450195 │
        │ CalibratedClassifierCV      │ 0.9760479041916168 │ 0.9760018220340847 │ 0.9755904096436046  │ NA                  │ 0.36926722526550293  │
        │ ComplementNB                │ 0.9760479041916168 │ 0.9752329192546583 │ 0.9754237510855159  │ NA                  │ 0.009947061538696289 │
        │ DecisionTreeClassifier      │ 0.8532934131736527 │ 0.8473956671194278 │ 0.8496464898940103  │ NA                  │ 0.34440088272094727  │
        │ DummyClassifier             │ 0.2155688622754491 │ 0.2                │ 0.07093596059113301 │ NA                  │ 0.005555868148803711 │
        │ ExtraTreeClassifier         │ 0.7275449101796407 │ 0.7253518459908658 │ 0.7255575847020816  │ NA                  │ 0.018934965133666992 │
        │ ExtraTreesClassifier        │ 0.9655688622754491 │ 0.9635363285903302 │ 0.9649837485086689  │ NA                  │ 1.2101161479949951   │
        │ GradientBoostingClassifier  │ 0.9550898203592815 │ 0.9526333887196529 │ 0.9539060578037555  │ NA                  │ 30.256237030029297   │
        │ KNeighborsClassifier        │ 0.938622754491018  │ 0.9370053693959814 │ 0.9367294513157219  │ NA                  │ 0.12071108818054199  │
        │ LinearSVC                   │ 0.9745508982035929 │ 0.974262691599302  │ 0.9740343976103922  │ NA                  │ 0.11713886260986328  │
        │ LogisticRegression          │ 0.968562874251497  │ 0.9668995859213251 │ 0.9678778814908909  │ NA                  │ 0.8916082382202148   │
        │ LogisticRegressionCV        │ 0.9715568862275449 │ 0.9708896757262861 │ 0.971147482393915   │ NA                  │ 37.82431483268738    │
        │ MLPClassifier               │ 0.9760479041916168 │ 0.9753381642512078 │ 0.9752912960666735  │ NA                  │ 30.700589656829834   │
        │ MultinomialNB               │ 0.9700598802395209 │ 0.9678795721187026 │ 0.9689200656860745  │ NA                  │ 0.01410818099975586  │
        │ NearestCentroid             │ 0.9520958083832335 │ 0.9499045135454718 │ 0.9515097876015481  │ NA                  │ 0.018617868423461914 │
        │ NuSVC                       │ 0.9670658682634731 │ 0.9656159420289855 │ 0.9669719954040374  │ NA                  │ 6.941549062728882    │
        │ PassiveAggressiveClassifier │ 0.9775449101796407 │ 0.9772388820754925 │ 0.9770812340935414  │ NA                  │ 0.05249309539794922  │
        │ Perceptron                  │ 0.9775449101796407 │ 0.9769254658385094 │ 0.9768161404324825  │ NA                  │ 0.030637741088867188 │
        │ RandomForestClassifier      │ 0.9625748502994012 │ 0.9605135542632081 │ 0.9624462948504477  │ NA                  │ 0.9921820163726807   │
        │ RidgeClassifier             │ 0.9775449101796407 │ 0.9769254658385093 │ 0.9769176825464448  │ NA                  │ 0.09582686424255371  │
        │ SGDClassifier               │ 0.9700598802395209 │ 0.9695007868373973 │ 0.969787370271274   │ NA                  │ 0.04686570167541504  │
        │ SVC                         │ 0.9715568862275449 │ 0.9703778467908902 │ 0.9713021262026043  │ NA                  │ 6.64256477355957     │
        └─────────────────────────────┴────────────────────┴────────────────────┴─────────────────────┴─────────────────────┴──────────────────────┘
    """

    def __init__(
        self,
        random_state: int = 13,
        classifiers: tt.List[str] = ["all"],
        classification_type: str = "binary",
        class_weights: bool = True,
        custom_metric: tt.Callable = None,
        custom_parameters: tt.List = None,
    ):
        self.random_state = random_state
        self.classifiers = classifiers
        self.results: tt.List = []
        self.classification_type = classification_type
        self.class_weights = class_weights
        self.custom_metric = custom_metric
        self.custom_parameters = custom_parameters

        # Set the global numpy random seed for reproducible results
        np.random.seed(self.random_state)

    @property
    def get_all_classifiers(self):
        """Get all the classifiers from Scikit-learn and filter them.

        Returns:
            filtered estimators.
        """
        estimators = dict(all_estimators(type_filter="classifier"))
        # filter the estimators
        filtered_estimators = {
            name: est for name, est in estimators.items() if name not in DO_NOT_APPLY
        }

        if self.classifiers == ["all"]:
            return filtered_estimators
        elif isinstance(self.classifiers, list):
            ud_estimators = {
                name: est
                for name, est in filtered_estimators.items()
                if name in self.classifiers
            }
            return ud_estimators

    def label_analysis(self, x_test, y_test):
        """Analyse the labels for classification and compute the class weights.

        Args:
            x_test: Training labels.
            y_test: Testing labels.

        Returns:
            Class weights of the labels.
        """
        if not isinstance(x_test, list) and not isinstance(y_test, list):
            return

        labels = x_test + y_test
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)

        unique_labeles = list(set(labels))

        logger.info("Computing class weights")
        class_weights_array = compute_class_weight(
            class_weight="balanced", classes=np.unique(encoded_labels), y=encoded_labels
        )

        class_weights = dict(zip(unique_labeles, class_weights_array.tolist()))

        return class_weights

    def calculate_matrix(self, true_labels, predictions):
        """Compute the metric for each of the classification estimator.

        Args:
            true_labels: True labels.
            predictions: Predicted labels.

        Returns:
            Dictonary of computed metrices.
        """
        accuracy = sklearn.metrics.accuracy_score(true_labels, predictions)
        balanced_accuracy = sklearn.metrics.balanced_accuracy_score(
            true_labels, predictions
        )
        if self.classification_type == "binary":
            f1_score = sklearn.metrics.f1_score(true_labels, predictions)
        elif self.classification_type == "multiclass":
            f1_score = sklearn.metrics.f1_score(
                true_labels, predictions, average="macro"
            )
        else:
            f1_score = "NA"

        classification_report = sklearn.metrics.classification_report(
            true_labels, predictions
        )
        confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, predictions)

        # custom metric
        # check if it hasattr y_true, y_pred and nothing else
        if self.custom_metric:
            if inspect.getfullargspec(self.custom_metric).args != ["y_true", "y_pred"]:
                logger.warning(
                    "Custom Metric should only have two arguments y_true and y_pred and it should return only one score. The signature of your custom mertic does not match the required signature. Ignoring the custom metric evaluation."
                )
                custom_score = "NA"
            else:
                custom_score = self.custom_metric(true_labels, predictions)
        else:
            custom_score = "NA"

        return {
            "Accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "f1_score": f1_score,
            "classification_report": classification_report,
            "confusion_matrix": confusion_matrix,
            "custom_metric_score": custom_score,
        }

    def _check_custom_parameters(self, parameters, model):
        """Check the custom user defined parameters.

        Example:
            parameter = {
                "name": "MultinomialNB",
                "parameters: {
                    "alpha": 1.0,
                    "fit_prior": True,
                    "class_prior": None
                }
            }

        Args:
            parameters: User defined parameters.
            model: Estimator for which the user defined the parameter.

        Raises:
            ValueError: If the parameters are not found in the estimator.

        Returns:
            Parameters to be applied to the estimator.
        """
        user_paramters = list(parameters["parameters"].keys())
        model_parameters = list(model.__dict__.keys())

        for user_parameter in user_paramters:
            if user_parameter not in model_parameters:
                raise ValueError(
                    f"The parameter {user_parameter} provided is not available for model"
                )

        return parameters["parameters"]

    def _check_and_convert_labels(self, y_train, y_test):
        """Encode the labels if not already encoded.

        Args:
            y_train: Training labels.
            y_test: Testing labels.

        Returns:
            Encoded Training labels and Testing labels
        """
        if not all(isinstance(y, int) for y in y_train) or not all(
            isinstance(y, int) for y in y_test
        ):
            logger.info("Label Encoding is required.")
            le = LabelEncoder()
            le.fit(y_train)
            encoded_train_labels = le.transform(y_train)
            encoded_test_labels = le.transform(y_test)

            return encoded_train_labels, encoded_test_labels
        else:
            return y_train, y_test

    def fit(self, x_train, x_test, y_train, y_test):
        """Train and evaluate the estimators for classification.

        Args:
            x_train: Training data.
            x_test: Testing data.
            y_train: Training labels.
            y_test: Testing labels.

        Raises:
            ValueError: If the estimator does not have a fit method.
            ValueError: If the estimator does not hae predict method.

        Returns:
            A list of results.
        """
        # get all the classifiers first
        class_weights = self.label_analysis(y_train, y_test)
        models = self.get_all_classifiers
        logger.info("Checking labels.")

        y_train, y_test = self._check_and_convert_labels(y_train, y_test)
        for name, estimator in track(models.items(), description="Training..."):
            print(f"  Training {name} estimator")
            model = estimator()

            # check if there are model parameters
            if self.custom_parameters:
                for params in self.custom_parameters:
                    if params.get("name") == name:
                        # check the custom parameters and create the model
                        user_parameters = self._check_custom_parameters(params, model)
                        model = estimator(**user_parameters)

            start = time.time()
            if not hasattr(model, "fit"):
                raise ValueError(
                    f"The esitmator {name} does not seem to have a fit method. Please report this bug on github."
                )

            # check if model accepts class weights
            model_args = model.__dict__
            logger.info(
                f"Estimator {name} has class weights argument, applying appropriate class weights."
            )
            if model_args.get("class_weight") and self.class_weights:
                model = estimator(class_weight=class_weights)

            model.fit(x_train, y_train)
            logger.info(f"Training of model completed for estimator {name}")
            if not hasattr(model, "predict"):
                raise ValueError(
                    f"The estimator {name} doeos not seem to have predict method. Please report this bug on github."
                )
            predictions = model.predict(x_test)
            logger.info(f"Evaluating {name}.")
            result = self.calculate_matrix(y_test, predictions)
            end = time.time()
            self.results.append(
                {
                    "name": name,
                    "accuracy": result["Accuracy"],
                    "balanced_accuracy": result["balanced_accuracy"],
                    "f1_score": result["f1_score"],
                    "custom_metric_score": result["custom_metric_score"],
                    "time": end - start,
                    "model": model,
                    "confusion_matrix": result["confusion_matrix"],
                    "classification_report": result["classification_report"],
                }
            )
        create_table(self.results, class_weights)
        return self.results
