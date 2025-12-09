import numpy as np
import pickle
from flask import Flask, request, jsonify
import tensorflow as tf  # Changed from tflite_runtime
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Global Variables ---
condition_ids = None
drug_ids = None
id_to_drug = None
model = None
input_details = None
output_details = None

def load_resources():
    """Load all required resources (mappings and model)"""
    global condition_ids, drug_ids, id_to_drug, model, input_details, output_details
    
    try:
        # Load mappings
        logger.info("🔄 Loading mappings...")
        with open('recommendation_mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        
        condition_ids = mappings['condition_ids']
        drug_ids = mappings['drug_ids']
        id_to_drug = mappings['id_to_drug']
        logger.info(f"✅ Mappings loaded: {len(condition_ids)} conditions, {len(drug_ids)} drugs")
        
        # Load TFLite model using TensorFlow
        logger.info("🔄 Loading TFLite model...")
        
        # Try multiple model loading approaches
        model_path = 'drug_recommendation_quantized.tflite'
        
        if not os.path.exists(model_path):
            # Try alternative paths
            possible_paths = [
                'drug_recommendation_quantized.tflite',
                './drug_recommendation_quantized.tflite',
                'models/drug_recommendation_quantized.tflite'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    logger.info(f"📁 Found model at: {path}")
                    break
            else:
                raise FileNotFoundError(f"Model file not found. Tried: {possible_paths}")
        
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"📊 Model info:")
        logger.info(f"   - Input shape: {input_details[0]['shape']}")
        logger.info(f"   - Input dtype: {input_details[0]['dtype']}")
        logger.info(f"   - Output shape: {output_details[0]['shape']}")
        
        return interpreter
        
    except Exception as e:
        logger.error(f"❌ Error loading resources: {str(e)}")
        raise

# Load resources at startup
try:
    interpreter = load_resources()
    logger.info("🚀 All resources loaded successfully!")
except Exception as e:
    logger.error(f"💥 Failed to load resources: {e}")
    # Don't exit - let the app start but it will fail on API calls
    interpreter = None

# --- Helper Function with Batching ---
def get_drug_recommendations(condition_name, top_k=5, batch_size=50):
    """Get top K drug recommendations using quantized model with batching"""
    global interpreter, condition_ids, drug_ids, id_to_drug
    
    if interpreter is None:
        raise RuntimeError("Model not loaded. Check server logs.")
    
    if condition_name not in condition_ids:
        available = list(condition_ids.keys())[:10]
        raise ValueError(f"Condition '{condition_name}' not found. Try: {available}")
    
    cond_id = condition_ids[condition_name]
    all_drug_ids = list(range(len(drug_ids)))
    total_drugs = len(all_drug_ids)
    
    all_predictions = []
    
    # Process in batches to save memory
    for i in range(0, total_drugs, batch_size):
        batch_drugs = all_drug_ids[i:i + batch_size]
        
        # Prepare batch inputs - check model's expected dtype
        input_dtype = input_details[0]['dtype']
        
        if input_dtype == np.float32:
            cond_array = np.array([cond_id] * len(batch_drugs), dtype=np.float32)
            drug_array = np.array(batch_drugs, dtype=np.float32)
        elif input_dtype == np.int32:
            cond_array = np.array([cond_id] * len(batch_drugs), dtype=np.int32)
            drug_array = np.array(batch_drugs, dtype=np.int32)
        else:
            # Default to float32
            cond_array = np.array([cond_id] * len(batch_drugs), dtype=np.float32)
            drug_array = np.array(batch_drugs, dtype=np.float32)
        
        # Set inputs
        interpreter.set_tensor(input_details[0]['index'], cond_array)
        interpreter.set_tensor(input_details[1]['index'], drug_array)
        
        # Run inference
        interpreter.invoke()
        
        # Get predictions
        batch_predictions = interpreter.get_tensor(output_details[0]['index'])
        all_predictions.extend(batch_predictions.flatten())
    
    # Convert to numpy array
    predictions = np.array(all_predictions)
    
    # Get top K recommendations
    top_indices = predictions.argsort()[-top_k:][::-1]
    
    recommendations = []
    for idx in top_indices:
        # Normalize score to 1-5 scale
        raw_score = float(predictions[idx])
        
        # Simple normalization (adjust based on your model's output range)
        if predictions.max() > predictions.min():
            normalized_score = 1 + 4 * ((raw_score - predictions.min()) / 
                                       (predictions.max() - predictions.min()))
        else:
            normalized_score = 3.0  # Default middle score
        
        recommendations.append({
            'drug': id_to_drug[idx],
            'drug_id': int(idx),
            'raw_score': float(raw_score),
            'effectiveness_score': round(normalized_score, 2),
            'effectiveness_display': f"{normalized_score:.2f}/5.00",
            'rank': len(recommendations) + 1
        })
    
    return recommendations

# --- API Routes ---
@app.route('/health', methods=['GET'])
def health_check():
    status = "healthy" if interpreter is not None else "unhealthy"
    model_status = "loaded" if interpreter is not None else "not loaded"
    
    response = {
        "status": status,
        "service": "Drug Recommendation API",
        "model": "quantized_tflite",
        "model_status": model_status,
        "available_conditions": len(condition_ids) if condition_ids else 0,
        "available_drugs": len(drug_ids) if drug_ids else 0,
        "python_version": os.sys.version,
        "tensorflow_version": tf.__version__
    }
    
    if interpreter is not None:
        response.update({
            "input_shape": str(input_details[0]['shape']),
            "output_shape": str(output_details[0]['shape'])
        })
    
    return jsonify(response)

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    EXPECTED JSON: 
    {
        "condition": "Pain",
        "top_k": 5  (optional, default 5)
    }
    """
    try:
        if interpreter is None:
            return jsonify({"error": "Model not loaded. Service unavailable."}), 503
        
        data = request.get_json()
        if not data or 'condition' not in data:
            return jsonify({"error": "Missing 'condition' field"}), 400
        
        condition = data['condition']
        top_k = data.get('top_k', 5)
        
        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 50:
            return jsonify({"error": "top_k must be an integer between 1 and 50"}), 400
        
        recommendations = get_drug_recommendations(condition, top_k)
        
        return jsonify({
            "status": "success",
            "condition": condition,
            "condition_id": condition_ids.get(condition),
            "recommendations": recommendations,
            "count": len(recommendations),
            "note": "Effectiveness scores are normalized to a 1-5 scale"
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.error(f"Error in /recommend: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/conditions', methods=['GET'])
def list_conditions():
    """Get all available medical conditions"""
    if condition_ids is None:
        return jsonify({"error": "Data not loaded"}), 503
    
    conditions = list(condition_ids.keys())
    return jsonify({
        "conditions": conditions,
        "count": len(conditions),
        "first_10": conditions[:10]  # Preview
    })

@app.route('/drugs', methods=['GET'])
def list_drugs():
    """Get all available drugs"""
    if drug_ids is None:
        return jsonify({"error": "Data not loaded"}), 503
    
    drugs = list(drug_ids.keys())
    return jsonify({
        "drugs": drugs,
        "count": len(drugs),
        "first_10": drugs[:10]  # Preview
    })

@app.route('/condition/<name>', methods=['GET'])
def get_condition_info(name):
    """Get information about a specific condition"""
    if condition_ids is None:
        return jsonify({"error": "Data not loaded"}), 503
    
    if name not in condition_ids:
        return jsonify({"error": f"Condition '{name}' not found"}), 404
    
    return jsonify({
        "condition": name,
        "condition_id": condition_ids[name],
        "exists": True
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API info"""
    endpoints = {
        "GET /health": "Health check and service status",
        "GET /conditions": "List all medical conditions",
        "GET /drugs": "List all drugs",
        "GET /condition/<name>": "Get info about specific condition",
        "POST /recommend": "Get drug recommendations (requires JSON)"
    }
    
    example_request = {
        "condition": "Pain",
        "top_k": 3
    }
    
    return jsonify({
        "message": "Drug Recommendation API",
        "version": "2.0",
        "description": "Machine learning API for drug recommendations based on medical conditions",
        "endpoints": endpoints,
        "example_request": example_request,
        "curl_example": "curl -X POST https://your-api.onrender.com/recommend -H 'Content-Type: application/json' -d '{\"condition\": \"Pain\", \"top_k\": 3}'"
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

# --- Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Starting Drug Recommendation API on port {port}")
    logger.info(f"📦 TensorFlow version: {tf.__version__}")
    logger.info(f"🐍 Python version: {os.sys.version}")
    
    # Verify resources are loaded
    if interpreter is not None and condition_ids is not None:
        logger.info("✅ All systems ready!")
    else:
        logger.warning("⚠️  Some resources failed to load. API may not function properly.")
    
    app.run(host='0.0.0.0', port=port, debug=False)