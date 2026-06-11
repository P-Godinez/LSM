from typing import Counter

import numpy as np
from model import get_modelGRU
from model import get_model

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.utils import to_categorical
from helpers import get_word_ids, get_sequences_and_labels
from constants import *
from collections import Counter

def training_model(model_path, epochs=500):
    word_ids = get_word_ids(WORDS_JSON_PATH ) # ['word1', 'word2', 'word3]
    
    sequences, labels = get_sequences_and_labels(word_ids)
    sequences = pad_sequences(sequences, maxlen=int(MODEL_FRAMES), padding='pre', truncating='post', dtype='float16')
    
    X = np.array(sequences)
    y = to_categorical(labels).astype(int) 
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=15, restore_best_weights=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = get_modelGRU(int(MODEL_FRAMES), len(word_ids)) ## CAMBIAR MODELO
    print("Clases:", word_ids)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Min:", np.min(X))
    print("Max:", np.max(X))
    print("Mean:", np.mean(X))

    print(Counter(labels))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=8, callbacks=[early_stopping])
    
    pred = model.predict(X_val, verbose=0)  ## PRINT UNA MÉTRICA EXTRA DE F1 PARA LA EVALUACIÓN DEL MODELO 
    yP = np.argmax(pred, axis = 1)
    yT = np.argmax(y_val, axis=1)
    f1 = f1_score(yT ,yP, average='macro')
    print(f"F1 Macro (Indicador mejor para datos no tan balaenceados): {f1:.6f}")
    
    model.summary()
    
    model.save(model_path)

if __name__ == "__main__":
    training_model(MODEL_PATH)
    