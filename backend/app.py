from flask import Flask, request, jsonify, send_file
from PIL import Image
import traceback
import pipeline

app = Flask(__name__)

cors_headers = { 
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET,PUT,POST,DELETE', 
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Credentials': 'true'
}

# Test endpoint
@app.route('/', methods=['GET', 'POST'])
def home():
    try:    
        return send_file(
            './last_processed.png',
            mimetype='image/png',
            as_attachment=True,
            download_name='processed_image.png'
        ), 200, cors_headers
    except Exception as e:
        return jsonify({'error': str(e)}), 500, cors_headers
    #return jsonify({'msg': 'Home'}), 200, cors_headers

@app.route('/process-image', methods=['POST'])
def process_image():

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
    
    # Controllo dello stato della pipeline: se non è pronta (init non completo), rigetto la richiesta
    #if not pipeline.ready:
    #    return jsonify({'error': 'Pipeline is not ready: please retry later'}), 500, cors_headers

    # Elaborazione dell'immagine
    try:    
        image = Image.open(file.stream)
        processed_image = pipeline.start(image, team1, team2)
        processed_image.save('./last_processed.png', 'PNG')
    except Exception as e:
        with open('app.log', 'w+') as f:
            traceback.print_exc(file=f)
        return jsonify({'error': str(e)}), 500, cors_headers
    
    return send_file(
        './last_processed.png',
        mimetype='image/png',
        as_attachment=True,
        download_name='processed_image.png'
    ), 200, cors_headers

if __name__ == '__main__':
    pipeline.init()
    app.run(debug=True, port=5000, host='0.0.0.0')
