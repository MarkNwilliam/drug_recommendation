import tensorflow as tf
import os

print("1. Checking files...")
if not os.path.exists('drug_recommendation_model.keras'):
    print("❌ ERROR: Model file not found!")
    exit()

print("✅ Found 'drug_recommendation_model.keras'")

print("\n2. Loading model...")
model = tf.keras.models.load_model('drug_recommendation_model.keras')
print("✅ Model loaded")

print("\n3. Quantizing model (reduces size 50-75%)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

print("\n4. Saving quantized model...")
with open('drug_recommendation_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("\n5. Checking sizes...")
original_size = os.path.getsize('drug_recommendation_model.keras') / (1024*1024)
quantized_size = os.path.getsize('drug_recommendation_quantized.tflite') / (1024*1024)

print(f"📊 Original model: {original_size:.2f} MB")
print(f"📊 Quantized model: {quantized_size:.2f} MB")
print(f"🎯 Reduction: {((original_size - quantized_size)/original_size)*100:.0f}%")

print("\n✅ DONE! Use 'drug_recommendation_quantized.tflite' on Render.")