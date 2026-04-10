# Anomaly Detection Webapp

A lightweight web interface for testing anomaly detection models on 96-bin physical detector data. 

I built this because testing edge cases via CLI or Jupyter notebooks gets tedious. This tool lets you manually "draw" a signal on a histogram using your mouse, runs it through a PyTorch model, and gives you immediate visual feedback on whether the signal looks normal or anomalous.

## How it works

1. **Draw:** Click and drag on the histogram to shape your 96-bin signal.
2. **Evaluate:** The backend normalizes the data (min-max scaling per histogram) and feeds it into the PyTorch model (e.g., an Autoencoder).
3. **Score:** Instead of simple MSE, it uses a Quantile Scorer to evaluate the reconstruction loss. You can adjust the `q` value directly in the UI to filter out standard noise.
4. **Feedback:** Returns a traffic-light status (Normal/Suspicious/Anomalous) and generates a residual heatmap so you can see exactly which bins caused the anomaly trigger.

## Quickstart

You'll need Python 3.x. I recommend using a virtual environment.

```bash
# Clone the repo
git clone https://github.com/jHrdy/anomaly_detection_webapp.git
cd anomaly_detection_webapp

# Install the required packages
pip install flask numpy torch matplotlib

# Run the server
python app.py
