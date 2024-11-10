from flask import Flask, request, jsonify, send_file
from PIL import Image
import traceback
import pipeline
from my_util import PIL_to_base64

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

@app.route('/process-image/start', methods=['POST'])
def process_image_start():

    if 'input_image' not in request.files:
        return jsonify({'error': 'No input image'}), 500, cors_headers
    
    if not request.form.get('team1'):
        return jsonify({'error': 'No team 1'}), 500, cors_headers
    
    if not request.form.get('team2'):
        return jsonify({'error': 'No team 2'}), 500, cors_headers
    
    input_image = request.files['input_image']
    team1 = request.form.get('team1')
    team2 = request.form.get('team2')
    
    if input_image.filename == '':
        return jsonify({'error': 'No selected file'}), 500, cors_headers

    try:
        image_base64 = PIL_to_base64(Image.open(input_image.stream))
        task = pipeline.start_new_task.delay(image_base64, team1, team2)
    except Exception as e:
        with open('app.log', 'w+') as f:
            traceback.print_exc(file=f)
        return jsonify({'error': str(e)}), 500, cors_headers

    return jsonify({'info': 'Your image processing request was enqueued successfully', 'task_id': task.id}), 202, cors_headers

@app.route('/process-image/update/<task_id>', methods=['GET'])
def process_image_update(task_id):
    task = pipeline.celery.AsyncResult(task_id)
    if task.state == 'PENDING':
        return jsonify({'status': 'Task is pending...'}), 202, cors_headers
    elif task.state == 'PROGRESS':
        return jsonify({'status': 'Task is in progress...'}), 202, cors_headers
    elif task.state == 'SUCCESS':
        return jsonify({'status': 'Task finished successfully, please call /finalize to get your result.'}), 200, cors_headers
    elif task.state == 'FAILURE':
        return jsonify({'status': 'Task failed for the following reason:\n' + str(task.traceback)}), 500, cors_headers
    else:
        return jsonify({'status': 'Unknown state...'}), 202, cors_headers

@app.route('/process-image/finalize/<task_id>', methods=['GET'])
def process_image_finalize(task_id):
    try:
        processed_image = pipeline.celery.AsyncResult(task_id).info
        processed_image.save(fp='./last_processed.png')
        return send_file(
            './last_processed.png',
            mimetype='image/png',
            as_attachment=True,
            download_name='processed_image.png'
        ), 200, cors_headers
    except Exception as e:
        with open('app.log', 'w+') as f:
            traceback.print_exc(file=f)
        return jsonify({'error': str(e)}), 500, cors_headers

if __name__ == '__main__':
    if not pipeline.INITIALIZED:
        pipeline.init_enviroment()
    app.run(debug=True, port=5000, host='0.0.0.0')
