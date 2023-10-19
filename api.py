from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)

with open('modelo_treinado.pkl', 'rb') as file:
    modelo_carregado = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')
        if not features:
            return jsonify({'error': 'Dados de entrada não fornecidos'}), 400
        prediction = modelo_carregado.predict([features])
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
