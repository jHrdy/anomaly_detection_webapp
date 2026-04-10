from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
from model_loader import load_model 
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
from scoring import QuantileScorer
from config import MODEL_PATH
matplotlib.use('Agg')

app = Flask(__name__)

if not MODEL_PATH:
    MODEL_PATH = 'model/ae_conv_v3.1_Wass_Reg_ep246.pth'
    
model = load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index_modern.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json()
        raw_data = np.array(payload['histogram'], dtype=np.float32)
        q_val = float(payload.get('q', 0.9))

        # scaling
        h_min, h_max = np.min(raw_data), np.max(raw_data)
        scaled_data = (raw_data - h_min) / (h_max - h_min) if h_max - h_min > 0 else np.zeros_like(raw_data)

        # inferrence
        input_tensor = torch.from_numpy(scaled_data)
        with torch.no_grad():
            reconstruction = model(input_tensor.reshape(1,1,96)).squeeze().numpy()

        # scoring
        scorer = QuantileScorer(q=q_val)
        current_score = scorer.score(scaled_data, reconstruction)

        # semaphore logic
        t_low = 0.27  # suspicious boundary
        t_high = 0.38 # anomalous bound

        if current_score >= t_high:
            status = "ANOMALOUS"
            color = "#e74c3c" 
        elif current_score >= t_low:
            status = "SUSPICIOUS"
            color = "#f39c12" 
        else:
            status = "NORMAL"
            color = "#2ecc71" 

        # heatmap
        heatmap_url = None
        if status != "NORMAL":
            heatmap_url = create_heatmap_string(scaled_data, reconstruction)

        return jsonify({
            'score': current_score,
            'status': status,
            'color': color,
            'heatmap': heatmap_url
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

def create_heatmap_string(original, recreation):
    # hm for web
    residuals = np.abs(original - recreation)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    
    cmap = plt.get_cmap('bwr')
    colors = cmap(residuals / (np.max(residuals) + 1e-6)) # color norm

    axes[0].bar(range(len(original)), original, color=colors)
    axes[0].set_title("Original (Heatmap Overlay)")
    
    axes[1].bar(range(len(recreation)), recreation, color='royalblue')
    axes[1].set_title("Reconstruction")

    plt.tight_layout()
    
    # prevod grafu na Base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig) # memory release (keep this)
    return f"data:image/png;base64,{plot_url}"

if __name__ == '__main__':
    app.run(debug=True)