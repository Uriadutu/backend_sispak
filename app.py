from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import os
import matplotlib
from flask_cors import CORS

app = Flask(__name__)
matplotlib.use('Agg')

@app.after_request
def add_headers(response):
    
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response


# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///water_status.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Model
class WaterData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    curah_hujan = db.Column(db.Float, nullable=False)
    ketinggian_air = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f'<WaterData {self.id} - {self.status}>'

# Create database
with app.app_context():
    db.create_all()

# Decision Tree Model
try:
    with open('decision_tree_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Model file 'decision_tree_model.pkl' not found. A new model will be created.")
    model = None

# Function to train model
def train_model():
    data = WaterData.query.all()
    if not data:
        raise ValueError("No data available to train the model.")
    
    # Extract data from database
    features = []
    labels = []
    status_mapping = {'Normal': 0, 'Waspada': 1, 'Siaga': 2, 'Awas': 3}
    for item in data:
        features.append([item.curah_hujan, item.ketinggian_air])
        labels.append(status_mapping[item.status])

    # Train Decision Tree model
    global model
    model = DecisionTreeClassifier()
    model.fit(features, labels)

    # Save model to file
    with open('decision_tree_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    return "Model trained and saved successfully."

def add_training_data(new_curah_hujan, new_ketinggian_air, new_status):
    # Load model yang sudah ada
    try:
        with open('decision_tree_model.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        # Jika model belum ada, buat model baru
        model = DecisionTreeClassifier()

    # Status mapping
    status_mapping = {'Normal': 0, 'Waspada': 1, 'Siaga': 2, 'Awas': 3}

    # Tambahkan data training baru
    new_features = np.array([[new_curah_hujan, new_ketinggian_air]])
    new_label = np.array([status_mapping[new_status]])

    # Gabungkan data training
    if hasattr(model, 'X_') and hasattr(model, 'y_'):
        # Jika model sudah memiliki data training sebelumnya
        X_train = np.vstack([model.X_, new_features])
        y_train = np.concatenate([model.y_, new_label])
    else:
        # Jika model baru atau belum memiliki data training
        X_train = new_features
        y_train = new_label

    # Train ulang model dengan data training yang sudah digabung
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Simpan model yang sudah diperbarui
    with open('decision_tree_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    return "Training data added and model updated successfully."

# Route baru untuk menambah data training
@app.route('/add-training-data', methods=['POST'])
def add_training():
    data = request.get_json()

    # Validasi input
    if not data.get('curah_hujan') or not data.get('ketinggian_air') or not data.get('status'):
        return jsonify({'message': 'curah_hujan, ketinggian_air, and status are required'}), 400

    try:
        result = add_training_data(
            data['curah_hujan'], 
            data['ketinggian_air'], 
            data['status']
        )
        return jsonify({'message': result}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500
        
# Function to generate decision tree image
def generate_decision_tree_image():
    plt.figure(figsize=(15, 10))
    plot_tree(
        model, 
        feature_names=['curah_hujan', 'ketinggian_air'], 
        class_names=['Normal', 'Waspada', 'Siaga', 'Awas'], 
        filled=True, 
        rounded=True
    )
    os.makedirs('static', exist_ok=True)  # Pastikan folder static ada
    output_path = os.path.join('static', 'decision_tree_image.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Penting untuk membersihkan sesi plotting
    return output_path

# Endpoint for prediction using decision tree
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Input validation
    if not data.get('curah_hujan') or not data.get('ketinggian_air'):
        return jsonify({'message': 'curah_hujan and ketinggian_air are required'}), 400

    if model is None:
        return jsonify({'message': 'Model is not trained. Please train the model first.'}), 400

    # Convert input features
    try:
        features = np.array([[data['curah_hujan'], data['ketinggian_air']]])
        
        # Prediksi menggunakan model
        prediction = model.predict(features)
        
        # Mapping prediction results
        status_mapping_reverse = {0: 'Normal', 1: 'Waspada', 2: 'Siaga', 3: 'Awas'}
        predicted_status = status_mapping_reverse[prediction[0]]

        # Hitung detail perhitungan dari model
        detail_perhitungan = {}
        
        # Dapatkan probabilitas prediksi dari model
        proba = model.predict_proba(features)[0]
        
        # Gunakan probabilitas untuk detail perhitungan
        for i, status in status_mapping_reverse.items():
            detail_perhitungan[status] = {
                'persentase_curah_hujan': round(proba[i] * 100, 2),
                'persentase_ketinggian_air': round(proba[i] * 100, 2),
                'skor_total': round(proba[i] * 100, 2)
            }

        # Generate decision tree image
        image_path = generate_decision_tree_image()

        # Send prediction results
        return jsonify({
            'status_akhir': predicted_status, 
            'detail_perhitungan': detail_perhitungan,
            'image_path': image_path
        }), 200

    except Exception as e:
        return jsonify({'message': str(e)}), 500
# Tampilkan Gambar
@app.route('/download-tree-image', methods=['GET'])
def download_tree_image():
    image_path = os.path.join('static', 'decision_tree_image.png')
    
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png', as_attachment=True)
    else:
        return jsonify({'message': 'Image not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
