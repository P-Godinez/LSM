import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from helpers import get_word_ids, get_sequences_and_labels
from constants import *


def genMatriz():
    wordIds= get_word_ids(WORDS_JSON_PATH)
    sequences, labels = get_sequences_and_labels(wordIds)

    X = pad_sequences(sequences ,maxlen = MODEL_FRAMES,padding = 'post',truncating  = 'post',dtype = 'float32')
    yTrue = np.array(labels)

    model = load_model(MODEL_PATH)

    yPredProb = model.predict(X)
    yPred = np.argmax(yPredProb, axis=1)
    cm = confusion_matrix(yTrue, yPred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=wordIds)
    disp.plot(cmap = "Blues")

    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    genMatriz()