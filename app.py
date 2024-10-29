from flask import Flask, request, jsonify, send_file
import io

app = Flask(__name__)

@app.route('/process-image', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    if 'team1' not in request.files:
        return jsonify({'error': 'No team 1'})    
    
    if 'team2' not in request.files:
        return jsonify({'error': 'No team 2'})
    
    file = request.files['file']
    team1 = request.files['team1']
    team2 = request.files['team2']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Esegui il processo di elaborazione dell'immagine qui
    image = Image.open(file.stream)
    processed_image = process_image(image)  # funzione di elaborazione personalizzata
    
    img_io = io.BytesIO()
    processed_image.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(
        img_io,
        mimetype='image/png',
        as_attachment=True,
        attachment_filename='processed_image.png'
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
