import numpy as np
import pickle
from flask import Flask, request, jsonify
import tflite_runtime.interpreter as tflite
import os

app = Flask(__name__)

# --- Load Your Mappings ---
print("🔄 Loading mappings...")
with open('recommendation_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

condition_ids = mappings['condition_ids']
drug_ids = mappings['drug_ids']
id_to_drug = mappings['id_to_drug']

print(f"✅ Mappings loaded: {len(condition_ids)} conditions, {len(drug_ids)} drugs")

# --- Load Quantized Model ---
print("🔄 Loading quantized model...")
interpreter = tflite.Interpreter(model_path='drug_recommendation_quantized.tflite')
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ Model loaded! Input shape: {input_details[0]['shape']}, Output shape: {output_details[0]['shape']}")

# --- Helper Function with Batching ---
def get_drug_recommendations(condition_name, top_k=5, batch_size=50):
    """Get top K drug recommendations using quantized model with batching"""
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
        
        # Prepare batch inputs (must match model's expected dtype - usually float32)
        cond_array = np.array([cond_id] * len(batch_drugs), dtype=np.float32)
        drug_array = np.array(batch_drugs, dtype=np.float32)
        
        # Set inputs for the quantized model
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
        effectiveness_score = float(predictions[idx]) * 5  # Convert to 1-5 scale
        recommendations.append({
            'drug': id_to_drug[idx],
            'effectiveness_score': effectiveness_score,
            'effectiveness_display': f"{effectiveness_score:.2f}/5.00",
            'rank': len(recommendations) + 1
        })
    
    return recommendations

# --- API Routes ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Drug Recommendation API",
        "model": "quantized_tflite",
        "available_conditions": len(condition_ids),
        "available_drugs": len(drug_ids),
        "model_size": "0.08 MB (91% reduced)"
    })

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
            "note": "Effectiveness scores are on a 1-5 scale"
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/conditions', methods=['GET'])
def list_conditions():
    """Get all available medical conditions"""
    conditions = list(condition_ids.keys())
    return jsonify({
        "conditions": conditions,
        "count": len(conditions),
        "first_10": conditions[:10]  # Preview
    })

@app.route('/drugs', methods=['GET'])
def list_drugs():
    """Get all available drugs"""
    drugs = list(drug_ids.keys())
    return jsonify({
        "drugs": drugs,
        "count": len(drugs),
        "first_10": drugs[:10]  # Preview
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API info"""
    return jsonify({
        "message": "Drug Recommendation API",
        "version": "1.0",
        "endpoints": {
            "GET /health": "Health check",
            "GET /conditions": "List all medical conditions",
            "GET /drugs": "List all drugs",
            "POST /recommend": "Get drug recommendations (requires JSON with 'condition')"
        },
        "example": {
            "curl": "curl -X POST https://your-api.onrender.com/recommend -H 'Content-Type: application/json' -d '{\"condition\": \"Pain\"}'"
        }
    })

# --- Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)