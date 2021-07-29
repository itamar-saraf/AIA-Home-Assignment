import pickle
from pathlib import Path


def load_model():
    """
    Read the model, count vectorizer, label encoder
    :return:
    """
    filename = 'finalized_model.sav'
    with open(filename, 'rb') as file:
        cv, le, clf = pickle.load(file)
    return cv, le, clf


def languageDetection(filePath: Path):
    """
    Detect language of the input file
    :param filePath: the path to he sample you want to detect
    :return: the language labelling
    """
    text = filePath.read_text()
    cv, le, clf = load_model()

    ngram = cv.transform([text]).toarray()
    y_pred = clf.predict(ngram)
    return le.inverse_transform(y_pred)
