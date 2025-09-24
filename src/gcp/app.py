import os
from flask import Flask, jsonify, request
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'lit-llmed-gcp',
        'version': '1.0.0'
    })

@app.route('/api/status')
def status():
    """Service status endpoint"""
    return jsonify({
        'service': 'lit-llmed-gcp',
        'status': 'running',
        'environment': os.getenv('ENV', 'development')
    })

@app.route('/api/process', methods=['POST'])
def process_request():
    """Basic processing endpoint for future functionality"""
    try:
        data = request.get_json()
        logger.info(f"Received request: {data}")
        
        return jsonify({
            'message': 'Request processed successfully',
            'received_data': data,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'error': 'Failed to process request',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)