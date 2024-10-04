import pickle
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__, template_folder='F:/workspace/python/proyectos/despliegues')

model_Tree, min_max_scaler, variables = pickle.load(open('F:/workspace/python/proyectos/despliegues/modeloreal.pkl', 'rb'))
print(variables)

@app.route('/')
def Home():
    return render_template('indexvideo.html')

@app.route('/predict', methods=['POST'])
def predict():
    videojuego = (request.form['videojuego'])
    edad = int(request.form['Edad'])
    sexo = (request.form['Sexo'])
    plataforma = (request.form['Plataforma'])
    consumidor_habitual = (request.form['Consumidor_habitual'])
    
    print("VOSOFOS",videojuego, edad, sexo, plataforma, consumidor_habitual)

    # Crear un DataFrame base con las características
    input_data = pd.DataFrame({
        'Edad': [edad],
        'videojuego': [videojuego],
        'Sexo': [sexo],
        'Plataforma': [plataforma],
        'Consumidor_habitual': [consumidor_habitual]
    })
   
    # Usar pd.get_dummies para codificar las variables categóricas
    input_data = pd.get_dummies(input_data, columns=['videojuego', 'Plataforma'], drop_first=False, dtype=int)
    input_data = pd.get_dummies(input_data, columns=['Sexo', 'Consumidor_habitual'], drop_first=True, dtype=int)
    print(input_data.head())
    
    #Asegurarse que las columnas del input sean las mismas que las del modelo
    input_data = input_data.reindex(columns=variables, fill_value=0)
    

    # Realizar la predicción
    prediction_tree = model_Tree.predict(input_data)
    final_prediction = prediction_tree  

    return render_template('indexvideo.html', prediction_text='El precio del videojuego es {}'.format(final_prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)