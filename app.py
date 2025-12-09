import numpy as np
import pickle
from flask import Flask, request, jsonify
import tensorflow as tf
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
interpreter = None
input_details = None
output_details = None

def load_resources():
    """Load all required resources (mappings and model)"""
    global condition_ids, drug_ids, id_to_drug, interpreter, input_details, output_details
    
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
        
        # Log detailed model info
        logger.info(f"✅ Model loaded successfully!")
        logger.info("📊 Detailed model info:")
        for i, inp in enumerate(input_details):
            logger.info(f"   Input {i}: shape={inp['shape']}, dtype={inp['dtype']}, name={inp.get('name', 'N/A')}")
        for i, out in enumerate(output_details):
            logger.info(f"   Output {i}: shape={out['shape']}, dtype={out['dtype']}")
        
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
    interpreter = None

# --- Helper Function with Batching ---
def get_drug_recommendations(condition_name, top_k=5, batch_size=50):
    """Get top K drug recommendations using quantized model with batching"""
    global interpreter, condition_ids, drug_ids, id_to_drug, input_details
    
    if interpreter is None:
        raise RuntimeError("Model not loaded. Check server logs.")
    
    if condition_name not in condition_ids:
        available = list(condition_ids.keys())[:10]
        raise ValueError(f"Condition '{condition_name}' not found. Try: {available}")
    
    cond_id = condition_ids[condition_name]
    all_drug_ids = list(range(len(drug_ids)))
    total_drugs = len(all_drug_ids)
    
    all_predictions = []
    
    # Check input shape requirements
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    logger.debug(f"Model expects input shape: {input_shape}, dtype: {input_dtype}")
    
    # Process in batches to save memory
    for i in range(0, total_drugs, batch_size):
        batch_drugs = all_drug_ids[i:i + batch_size]
        batch_len = len(batch_drugs)
        
        # Prepare batch inputs based on model's expected shape
        if len(input_shape) == 2:
            # Model expects 2D input: [batch_size, 1] or [batch_size, features]
            if input_shape[-1] == 1:
                # Shape like [batch_size, 1]
                cond_array = np.full((batch_len, 1), cond_id, dtype=input_dtype)
                drug_array = np.array(batch_drugs, dtype=input_dtype).reshape(-1, 1)
            else:
                # General 2D case
                cond_array = np.full((batch_len, input_shape[1]), cond_id, dtype=input_dtype)
                drug_array = np.array(batch_drugs, dtype=input_dtype).reshape(-1, 1)
        elif len(input_shape) == 1 and input_shape[0] == -1:
            # Variable 1D input
            cond_array = np.full(batch_len, cond_id, dtype=input_dtype)
            drug_array = np.array(batch_drugs, dtype=input_dtype)
        else:
            # Default: assume 2D input
            cond_array = np.full((batch_len, 1), cond_id, dtype=input_dtype)
            drug_array = np.array(batch_drugs, dtype=input_dtype).reshape(-1, 1)
        
        # Debug logging
        logger.debug(f"Batch {i//batch_size}: cond_array shape={cond_array.shape}, drug_array shape={drug_array.shape}")
        
        # Set inputs - verify which input is which
        try:
            interpreter.set_tensor(input_details[0]['index'], cond_array)
            interpreter.set_tensor(input_details[1]['index'], drug_array)
            
            # Run inference
            interpreter.invoke()
            
            # Get predictions
            batch_predictions = interpreter.get_tensor(output_details[0]['index'])
            all_predictions.extend(batch_predictions.flatten())
            
        except ValueError as e:
            # Try swapping inputs if the first attempt fails
            logger.warning(f"First attempt failed: {e}. Trying swapped inputs...")
            try:
                interpreter.set_tensor(input_details[0]['index'], drug_array)
                interpreter.set_tensor(input_details[1]['index'], cond_array)
                interpreter.invoke()
                batch_predictions = interpreter.get_tensor(output_details[0]['index'])
                all_predictions.extend(batch_predictions.flatten())
            except Exception as e2:
                logger.error(f"Both input orders failed: {e2}")
                raise RuntimeError(f"Model input error: {e2}")
    
    # Convert to numpy array
    predictions = np.array(all_predictions)
    
    # Get top K recommendations
    if len(predictions) < top_k:
        top_k = len(predictions)
    
    top_indices = predictions.argsort()[-top_k:][::-1]
    
    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        raw_score = float(predictions[idx])
        
        # Normalize score to 1-5 scale if we have multiple predictions
        if len(predictions) > 1 and predictions.max() > predictions.min():
            normalized_score = 1 + 4 * ((raw_score - predictions.min()) / 
                                       (predictions.max() - predictions.min()))
        else:
            normalized_score = 3.0
        
        recommendations.append({
            'drug': id_to_drug[idx],
            'drug_id': int(idx),
            'raw_score': round(raw_score, 4),
            'effectiveness_score': round(normalized_score, 2),
            'effectiveness_display': f"{normalized_score:.2f}/5.00",
            'rank': rank
        })
    
    logger.info(f"Generated {len(recommendations)} recommendations for condition '{condition_name}'")
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
        "python_version": os.sys.version.split()[0],
        "tensorflow_version": tf.__version__
    }
    
    if interpreter is not None and input_details is not None:
        response.update({
            "input_shapes": [str(inp['shape']) for inp in input_details],
            "output_shape": str(output_details[0]['shape']),
            "input_dtypes": [str(inp['dtype']) for inp in input_details]
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
        "conditions": sorted(conditions),
        "count": len(conditions),
        "first_10": sorted(conditions)[:10]
    })

@app.route('/drugs', methods=['GET'])
def list_drugs():
    """Get all available drugs"""
    if drug_ids is None:
        return jsonify({"error": "Data not loaded"}), 503
    
    drugs = list(drug_ids.keys())
    return jsonify({
        "drugs": sorted(drugs),
        "count": len(drugs),
        "first_10": sorted(drugs)[:10]
    })

@app.route('/condition/<name>', methods=['GET'])
def get_condition_info(name):
    """Get information about a specific condition"""
    if condition_ids is None:
        return jsonify({"error": "Data not loaded"}), 503
    
    if name not in condition_ids:
        # Try case-insensitive search
        name_lower = name.lower()
        matches = [c for c in condition_ids.keys() if c.lower() == name_lower]
        if matches:
            name = matches[0]
        else:
            return jsonify({"error": f"Condition '{name}' not found"}), 404
    
    return jsonify({
        "condition": name,
        "condition_id": condition_ids[name],
        "exists": True
    })

@app.route('/debug/model', methods=['GET'])
def debug_model():
    """Debug endpoint to see model details"""
    if interpreter is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    info = {
        "input_details": [],
        "output_details": []
    }
    
    for i, inp in enumerate(input_details):
        info["input_details"].append({
            "index": i,
            "shape": inp['shape'].tolist() if hasattr(inp['shape'], 'tolist') else inp['shape'],
            "dtype": str(inp['dtype']),
            "name": inp.get('name', 'N/A')
        })
    
    for i, out in enumerate(output_details):
        info["output_details"].append({
            "index": i,
            "shape": out['shape'].tolist() if hasattr(out['shape'], 'tolist') else out['shape'],
            "dtype": str(out['dtype'])
        })
    
    return jsonify(info)

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API info"""
    endpoints = {
        "GET /health": "Health check and service status",
        "GET /conditions": "List all medical conditions",
        "GET /drugs": "List all drugs",
        "GET /condition/<name>": "Get info about specific condition",
        "POST /recommend": "Get drug recommendations (requires JSON)",
        "GET /debug/model": "Debug model info (development only)"
    }
    
    example_request = {
        "condition": "Pain",
        "top_k": 3
    }
    
    return jsonify({
        "message": "Drug Recommendation API",
        "version": "2.1",
        "description": "Machine learning API for drug recommendations based on medical conditions",
        "endpoints": endpoints,
        "example_request": example_request,
        "curl_example": "curl -X POST /recommend -H 'Content-Type: application/json' -d '{\"condition\": \"Pain\", \"top_k\": 3}'",
        "status": "running" if interpreter is not None else "initializing"
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

# --- App Startup Logic ---
def initialize_app():
    """Initialize the application"""
    logger.info(f"📦 TensorFlow version: {tf.__version__}")
    logger.info(f"🐍 Python version: {os.sys.version}")
    
    # Verify resources are loaded
    if interpreter is not None and condition_ids is not None:
        logger.info("✅ All systems ready!")
        return True
    else:
        logger.warning("⚠️  Some resources failed to load. API may not function properly.")
        return False

# Initialize when module loads
app_ready = initialize_app()

# This block only runs if you execute: python app.py
# (Render uses gunicorn which imports the module, so this doesn't run in production)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Starting Drug Recommendation API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)