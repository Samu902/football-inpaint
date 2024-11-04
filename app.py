from flask import Flask, request, jsonify, send_file
from PIL import Image
import io

app = Flask(__name__)

cors_headers = { 
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET,PUT,POST,DELETE', 
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Credentials': 'true'
}

# Test endpoint
@app.route('/', methods=['GET'])
def home():
    return jsonify({'msg': 'Home'}), 200, cors_headers

@app.route('/process-image', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 500, cors_headers
    
    if not request.form.get('team1'):
        return jsonify({'error': 'No team 1'}), 500, cors_headers
    
    if not request.form.get('team2'):
        return jsonify({'error': 'No team 2'}), 500, cors_headers
    
    file = request.files['file']
    team1 = request.form.get('team1')
    team2 = request.form.get('team2')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 500, cors_headers
    
    # Elaborazione dell'immagine
    image = Image.open(file.stream)
    
    #processed_image = process_image(image)  # funzione di elaborazione personalizzata
    processed_image = image
    
    processed_image.save('./last_processed.png', 'PNG')
    
    return send_file(
        './last_processed.png',
        mimetype='image/png',
        as_attachment=True,
        attachment_filename='processed_image.png'
    ), 200, cors_headers

if __name__ == '__main__':
    app.run(debug=True, port=5000)
