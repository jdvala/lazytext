# LazyText

![lazy](docs/sloth.png)


[![lazytext](https://github.com/jdvala/lazytext/actions/workflows/main.yml/badge.svg)](https://github.com/jdvala/lazytext/actions/workflows/main.yml)
[![Documentation](https://github.com/jdvala/lazytext/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/jdvala/lazytext/actions/workflows/pages/pages-build-deployment)
[![Code Coverage](https://codecov.io/gh/jdvala/lazytext/branch/master/graph/badge.svg)](https://codecov.io/gh/jdvala/lazytext)
[![Downloads](https://pepy.tech/badge/lazytext/month)](https://pepy.tech/project/lazytext)


LazyText is inspired b the idea of [lazypredict](https://github.com/shankarpandala/lazypredict), a library which helps build a lot of basic mpdels without much code. LazyText is for text what lazypredict is for numeric data.

* Free Software: MIT licence


## Installation

To install LazyText

`pip install lazytext`


## Usage

To use lazytext import in your project as

`from lazytext.supervised import LazyTextPredict`


## Text Classification

Text classification on BBC News article classification.

```python
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
| Classes             | Weights              |
|--------------------:|---------------------:|
| tech                | 0.8725490196078431   |
| politics            | 1.1528497409326426   |
| sport               | 1.0671462829736211   |
| entertainment       | 0.8708414872798435   |
| business            | 1.1097256857855362   |

 Result Analysis
| Model                         | Accuracy            | Balanced Accuracy   | F1 Score            | Custom Metric Score | Time Taken          |
| ----------------------------: | -------------------:| -------------------:| -------------------:| -------------------:| -------------------:|
| AdaBoostClassifier            | 0.7260479041916168  | 0.717737172132769   | 0.7248335989941609  | NA                  | 1.829047679901123   |
| BaggingClassifier             | 0.8817365269461078  | 0.8796633962363677  | 0.8814695332332374  | NA                  | 3.5215072631835938  |
| BernoulliNB                   | 0.9535928143712575  | 0.9505929193425733  | 0.9533647387436917  | NA                  | 0.020041465759277344|
| CalibratedClassifierCV        | 0.9760479041916168  | 0.9760018220340847  | 0.9755904096436046  | NA                  | 0.4990670680999756  |
| ComplementNB                  | 0.9760479041916168  | 0.9752329192546583  | 0.9754237510855159  | NA                  | 0.013598203659057617|
| DecisionTreeClassifier        | 0.8532934131736527  | 0.8473956671194278  | 0.8496464898940103  | NA                  | 0.478792667388916   |
| DummyClassifier               | 0.2155688622754491  | 0.2                 | 0.07093596059113301 | NA                  | 0.008046865463256836|
| ExtraTreeClassifier           | 0.7275449101796407  | 0.7253518459908658  | 0.7255575847020816  | NA                  | 0.026398658752441406|
| ExtraTreesClassifier          | 0.9655688622754491  | 0.9635363285903302  | 0.9649837485086689  | NA                  | 1.6907336711883545  |
| GradientBoostingClassifier    | 0.9565868263473054  | 0.9543725191544354  | 0.9554606292723953  | NA                  | 39.16400766372681   |
| KNeighborsClassifier          | 0.938622754491018   | 0.9370053693959814  | 0.9367294513157219  | NA                  | 0.14803171157836914 |
| LinearSVC                     | 0.9745508982035929  | 0.974262691599302   | 0.9740343976103922  | NA                  | 0.10053229331970215 |
| LogisticRegression            | 0.968562874251497   | 0.9668995859213251  | 0.9678778814908909  | NA                  | 2.9565982818603516  |
| LogisticRegressionCV          | 0.9715568862275449  | 0.9708896757262861  | 0.971147482393915   | NA                  | 109.64091444015503  |
| MLPClassifier                 | 0.9760479041916168  | 0.9753381642512078  | 0.9752912960666735  | NA                  | 35.64296746253967   |
| MultinomialNB                 | 0.9700598802395209  | 0.9678795721187026  | 0.9689200656860745  | NA                  | 0.024427413940429688|
| NearestCentroid               | 0.9520958083832335  | 0.9499045135454718  | 0.9515097876015481  | NA                  | 0.024636268615722656|
| NuSVC                         | 0.9670658682634731  | 0.9656159420289855  | 0.9669719954040374  | NA                  | 8.287142515182495   |
| PassiveAggressiveClassifier   | 0.9775449101796407  | 0.9772388820754925  | 0.9770812340935414  | NA                  | 0.10332632064819336 |
| Perceptron                    | 0.9775449101796407  | 0.9769254658385094  | 0.9768161404324825  | NA                  | 0.07216000556945801 |
| RandomForestClassifier        | 0.9625748502994012  | 0.9605135542632081  | 0.9624462948504477  | NA                  | 1.2427525520324707  |
| RidgeClassifier               | 0.9775449101796407  | 0.9769254658385093  | 0.9769176825464448  | NA                  | 0.17272400856018066 |
| SGDClassifier                 | 0.9700598802395209  | 0.9695007868373973  | 0.969787370271274   | NA                  | 0.13134551048278809 |
| SVC                           | 0.9715568862275449  | 0.9703778467908902  | 0.9713021262026043  | NA                  | 8.388679027557373   |
```

Result of each estimator is stored in `models` which is a list and each trained estimator is also returned which can be used further for analysis.

`confusion matrix` and `classification reports` are also part of the `models` if they are needed.


```python

print(models[0])
{
    'name': 'AdaBoostClassifier',
    'accuracy': 0.7260479041916168,
    'balanced_accuracy': 0.717737172132769,
    'f1_score': 0.7248335989941609,
    'custom_metric_score': 'NA',
    'time': 1.829047679901123,
    'model': AdaBoostClassifier(),
    'confusion_matrix': array([
        [ 89,   5,  12,  35,   3],
        [  8,  58,   5,  44,   0],
        [  5,   2, 108,  10,   1],
        [  5,   7,   5, 138,   2],
        [ 25,   5,   1,   3,  92]]),
 'classification_report':
 """
            precision    recall  f1-score   support
        0       0.67      0.62      0.64       144
        1       0.75      0.50      0.60       115
        2       0.82      0.86      0.84       126
        3       0.60      0.88      0.71       157
        4       0.94      0.73      0.82       126
 accuracy                           0.73       668
 macro avg       0.76      0.72     0.72       668
 weighted avg    0.75      0.73     0.72       668'}


```

### Custom metrics
LazyText also support custom metric for evaluation, this metric can be set up like following

```python
from lazytext.supervised import LazyTextPredict
# Custom metric
def my_custom_metric(y_true, y_pred):

    ...do your stuff

    return score


lazy_text = LazyTextPredict(custom_metric=my_custom_metric)
lazy_text.fit(X_train, X_test, y_train, y_test)
```

> If the signature of the custom metric function does not match with what is given above, then even though the custom metric is provided, it will be ignored.

### Custom model parameters

LazyText also support providing parameters to the esitmators. For this just provide a dictornary of the parameters as shown below and those following arguments will be applied to the desired estimator.

In the following example I want to apply/change the default parameters of `SVC` classifier.

> LazyText will fit all the models but only change the default parameters for SVC in the following case.

```python
from lazytext.supervisd
custom_parameters = [
    {
        "name": "SVC",
        "parameters": {
            "C": 0.5,
            "kernel": 'poly',
            "degree": 5
        }
    }
]


l = LazyTextPredict(
    classification_type="multiclass",
    custom_parameters=custom_parameters
    )
l.fit(x_train, x_test, y_train, y_test)
```
