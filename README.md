# INSTRUCCIONES DE USO
1. Correr *capture_samples.py* para hacer una grabación de muestras.
2. Correr *normalize_samples.py* para asegurar que todas las muestras tengan la misma cantidad de frames.
3. Correr *create_keypoints.py* para generar los archivos h.5
4. Correr *training_model.py* para entrenar al modelo. NOTA: Se puede cambiar entre funciones getModel() y getModelGRU() para cambiar la estructura ya sea LSTM o GRU.
5. Correr *evaluate_model.py* para realizar pruebas.
6. Correr *main.py*

El presente es un proyecto académico con el fin de entrenar una red neural para la detección de señas de LSM (lenguaje de señas mexicano).  

NOTA: Para la realización del proyecto se hizo uso con un repositorio público: https://github.com/ronvidev/modelo_lstm_lsp/tree/main. Se debe de mencionar también que se detectó en el código original un alto índice de uso de IA generativa.   

# Implementaciones significante

Para mejorar el proyecto original:  
    1. **Aplicación de ing. de características**  
        Se hizo una selección de variables relavantes por medio de la eliminación de los landmarks faciales y de pose que originalmente se consideraban en el modelo. Esta decisión se tomo debido a que todos los puntos faciales aportaban un valor nulo para la selección de señas.   
        Con esté cambio el modelo pasó de 1662 características a 126, haciendo así el modelo mucho más sencillo. En la practica, se observó un aumento significativo en la métrica de "accuracy" (+0.3)
        Los cambios pertinentes se pueden observar en *constants.py* en la cual pasó de LENGTH_KEYPOINTS = 1662 a LENGTH_KEYPOINTS = 126
        Asimismo, en *helpers.py* se rehizo la función *extract_keypoints()* para que unicamente considerara los puntos de la mano derecha e izquierda.      
    2. **Comparación con GRU**  
        Con el fin de hacer una rápida compración con otros modelos, se implementó en *model.py* la función *get_modelGRU()* la cual es bastante similar a la función original *get_model()* aunque más sencilla, adaptada a la estrctura GRU y con menos neuronas así como un menor dropout.  
        Sin embargo, los modelos resultaron tener un desempeño muy similar y con muy ligeras variaciones. 

