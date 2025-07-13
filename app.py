# app.py - MINIMAL TEST VERSION
# Comment out everything except basic Flask to test deployment

from flask import Flask, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({
        'status': 'working',
        'message': 'Vestara AI Platform - Basic Test',
        'platform': 'Railway',
        'version': '1.0.0-test'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'Flask app is running!'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
