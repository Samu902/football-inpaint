from flask import Flask, request, jsonify, send_file
import pipeline
from my_util import zip_bytes_to_base64, log_exc_to_file

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
            './pytorch_lora_weights.safetensors',
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='pytorch_lora_weights.safetensors'
        ), 200, cors_headers
    except Exception as e:
        log_exc_to_file()
        return jsonify({'error': str(e)}), 500, cors_headers

@app.route('/process-lora/start', methods=['POST'])
def process_lora_start():

    if 'input_file' not in request.files:
        return jsonify({'error': 'No input .zip file'}), 500, cors_headers
    
    if not request.form.get('team'):
        return jsonify({'error': 'No team'}), 500, cors_headers
    
    if not request.form.get('steps'):
        return jsonify({'error': 'No steps'}), 500, cors_headers
    
    input_file = request.files['input_file']
    team = request.form.get('team')
    steps = int(request.form.get('steps'))
    
    if input_file.filename == '':
        return jsonify({'error': 'No selected file'}), 500, cors_headers

    try:
        zip_b64 = zip_bytes_to_base64(input_file.stream.read())
        task = pipeline.start_new_task.delay(zip_b64, team, steps)
    except Exception as e:
        log_exc_to_file()
        return jsonify({'error': str(e)}), 500, cors_headers

    return jsonify({'info': 'Your lora training request was enqueued successfully', 'task_id': task.id}), 202, cors_headers

@app.route('/process-lora/update/<task_id>', methods=['GET'])
def process_lora_update(task_id):
    try:
        task = pipeline.celery.AsyncResult(task_id)
        if task.state == 'PENDING':
            return jsonify({'status': 'Task is pending...'}), 202, cors_headers
        elif task.state == 'PROGRESS':
            return jsonify({'status': 'Task is in progress...'}), 202, cors_headers
        elif task.state == 'SUCCESS':
            return jsonify({'status': 'Task finished successfully, please call /finalize to get your result.'}), 200, cors_headers
        elif task.state == 'FAILURE':
            with open('app.log', 'w+') as f:
                f.write(task.traceback)
            return jsonify({'status': 'Task failed for the following reason', 'error': str(task.traceback)}), 500, cors_headers
        else:
            return jsonify({'status': 'Unknown state...'}), 202, cors_headers
    except Exception as e:
        log_exc_to_file()
        return jsonify({'error': str(e)}), 500, cors_headers

@app.route('/process-lora/finalize/<task_id>', methods=['GET'])
def process_lora_finalize(task_id):
    try:
        return send_file(
            './pytorch_lora_weights.safetensors',
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='pytorch_lora_weights.safetensors'
        ), 200, cors_headers
    except Exception as e:
        log_exc_to_file()
        return jsonify({'error': str(e)}), 500, cors_headers

if __name__ == '__main__':
    app.run(debug=True, port=5002, host='0.0.0.0')
