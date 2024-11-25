from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from prediction import EmergencyPredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the predictor
try:
    predictor = EmergencyPredictor(
        model_path='models/saved_models/emergency_classifier_20241124_010648.pth'
    )
    logger.info("Emergency predictor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize predictor: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    """
    GET endpoint for predicting emergency level from tweet(s)
    
    Expected JSON body:
    {
        "tweets": "string" or ["string", "string", ...]
    }
    """
    try:
        # Get JSON data from request body
        data = request.get_json()
        
        if not data or 'tweets' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Request must include "tweets" field'
            }), 400
            
        # Handle both single tweet and list of tweets
        tweets = data['tweets']
        if isinstance(tweets, str):
            tweets = [tweets]
        elif not isinstance(tweets, list):
            return jsonify({
                'status': 'error',
                'message': '"tweets" must be either a string or an array of strings'
            }), 400
        
        # Make predictions
        predictions = predictor.predict(tweets)
        
        return jsonify({
            'status': 'success',
            'predictions': predictions
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)