import pickle
from flask import Flask,render_template,request
import pandas as pd
import os
#doc + tab (poner cuerpo html por defecto)


app = Flask(__name__)



archivo_modelo = os.path.join(os.path.dirname(__file__), 'rf_model.pkl')
model = pickle.load(open(archivo_modelo, 'rb'))

# Diccionario para almacenar las entradas de los clientes
diccionario = {}

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predecir", methods=['POST'])
def predecir():
    CLIENTE = str(request.form['cliente'])
    DESEMBOLSO = int(request.form['desembolso'])
    region = str(request.form['region'])
    sector = str(request.form['sector'])
    tamaño_empresa = str(request.form['TamañoEmpresa'])
    PLAZO_MEDIO = int(request.form['Plazo'])
    riesgo = str(request.form['riesgo'])
    linea_negocio = str(request.form['linea'])

    # Escalar las características que necesitas
    PLAZO_MEDIO_scaled = (PLAZO_MEDIO - 718.875402) / 529.194451
    DESEMBOLSO_scaled = (DESEMBOLSO - 1263.321038) / 1134.438532

    # Crear el DataFrame con las características escaladas y otras características
    test = pd.DataFrame({
        'PLAZO_MEDIO': [PLAZO_MEDIO_scaled],
        'DESEMBOLSO': [DESEMBOLSO_scaled],
        'REGIONAL_BOGOTA': [1 if region == "Bogotá" else 0],
        'REGIONAL_BUCARAMANGA': [1 if region == "Bucaramanga" else 0],
        'REGIONAL_CALI': [1 if region == "Cali" else 0],
        'REGIONAL_MEDELLIN': [1 if region == "Medellín" else 0],
        'SECTOR2_COMUNICACION': [1 if sector == "Comunicación" else 0],
        'SECTOR2_CONSTRUCCION': [1 if sector == "Construcción" else 0],
        'SECTOR2_FINANCIERO': [1 if sector == "Financiero" else 0],
        'SECTOR2_INMOBILIARIO': [1 if sector == "Inmobiliario" else 0],
        'SECTOR2_MANUFACTURA': [1 if sector == "Manufactura" else 0],
        'SECTOR2_MINERIA': [1 if sector == "Minería" else 0],
        'SECTOR2_OTRO SECTOR ECONOMICO': [1 if sector == "Otro" else 0],
        'SECTOR2_SERVICIOS': [1 if sector == "Servicios" else 0],
        'SECTOR2_TRANSPORTE': [1 if sector == "Transporte" else 0],
        'CARTERA_FACTORING EN FIRME': [1 if linea_negocio == "Factoring" else 0],
        'CARTERA_LEASING': [1 if linea_negocio == "Leasing" else 0],
        'CARTERA_OTRA LINEA DE NEGOCIO': [1 if linea_negocio == "Otro" else 0],
        'SEGMENTOMR_Mediana Empresa': [1 if tamaño_empresa == "Mediana Empresa" else 0],
        'SEGMENTOMR_Pequeña Empresa': [1 if tamaño_empresa == "Pequeña Empresa" else 0],
        'SEGMENTOMR_Persona Natural': [1 if tamaño_empresa == "Persona Natural" else 0],
        'NIVEL_RIESGO_2.Riesgo Alto': [1 if riesgo == "Riesgo Alto" else 0],
        'NIVEL_RIESGO_3.Riesgo Moderado': [1 if riesgo == "Riesgo Moderado" else 0],
        'NIVEL_RIESGO_4.Riesgo Bajo': [1 if riesgo == "Riesgo Bajo" else 0]
    })

    prediccion = model.predict_proba(test)
    output = round(prediccion[0, 1], 4)

    # Agregar las entradas del cliente al diccionario
    diccionario[CLIENTE] = {
        'DESEMBOLSO': DESEMBOLSO,
        'PLAZO_MEDIO': PLAZO_MEDIO,
        'REGION': region,
        'SECTOR': sector,
        'TAMAÑO_EMPRESA': tamaño_empresa,
        'RIESGO': riesgo,
        'LINEA_NEGOCIO': linea_negocio,
        'RATING': output
    }
    return render_template('index.html', prediccion_texto=f'El RATING INTERNO de Riesgo para el cliente {CLIENTE} es: {output}',riesgo=f'EL RATING DEL BURÓ es {riesgo}')

@app.route("/historial", methods=['GET'])
def historial():
    return str(diccionario)

if __name__=="__main__":
    app.run(debug=True,port=5000)





