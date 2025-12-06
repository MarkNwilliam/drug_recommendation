from flask import Flask, request, jsonify
import pickle
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# --- Load Your Models ---
print("🔄 Loading models...")
recommendation_model = tf.keras.models.load_model('drug_recommendation_model.keras')
with open('recommendation_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

# Extract mappings
condition_ids = mappings['condition_ids']
drug_ids = mappings['drug_ids']
id_to_drug = mappings['id_to_drug']

print("✅ Models loaded!")

# --- Helper Functions ---
def get_drug_recommendations(condition_name, top_k=5):
    """Get top K drug recommendations for a medical condition"""
    if condition_name not in condition_ids:
        available = list(condition_ids.keys())[:10]
        raise ValueError(f"Condition '{condition_name}' not found. Try: {available}")
    
    cond_id = condition_ids[condition_name]
    
    # Predict ratings for ALL drugs (this is key!)
    all_drug_ids = list(range(len(drug_ids)))  # [0, 1, 2, ..., N_drugs-1]
    cond_array = np.array([cond_id] * len(all_drug_ids))
    drug_array = np.array(all_drug_ids)
    
    # Model expects TWO inputs: [cond_array, drug_array]
    predictions = recommendation_model.predict([cond_array, drug_array], verbose=0)
    
    # Get top K recommendations
    top_indices = predictions.flatten().argsort()[-top_k:][::-1]
    
    recommendations = []
    for idx in top_indices:
        effectiveness_score = float(predictions[idx][0]) * 5  # Convert to 1-5 scale
        recommendations.append({
            'drug': id_to_drug[idx],
            'effectiveness_score': effectiveness_score,
            'effectiveness_display': f"{effectiveness_score:.2f}/5.00"
        })
    
    return recommendations

# --- API Endpoints ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "available_conditions": len(condition_ids),
        "available_drugs": len(drug_ids)
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
        
        # Get recommendations
        recommendations = get_drug_recommendations(condition, top_k)
        
        return jsonify({
            "status": "success",
            "condition": condition,
            "recommendations": recommendations,
            "count": len(recommendations)
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/conditions', methods=['GET'])
def list_conditions():
    """Get all available medical conditions"""
    return jsonify({
        "conditions": list(condition_ids.keys()),
        "count": len(condition_ids)
    })

@app.route('/drugs', methods=['GET'])
def list_drugs():
    """Get all available drugs"""
    return jsonify({
        "drugs": list(drug_ids.keys()),
        "count": len(drug_ids)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)