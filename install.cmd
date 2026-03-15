python -m venv audioai
audioai\Scripts\activate
pip install torch torchaudio librosa numpy soundfile onnx onnxruntime tqdm
pip uninstall torchcodec -y
pip uninstall torch torchaudio -y
pip install torch==2.9.1+cpu torchvision==0.24.1+cpu torchaudio==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu