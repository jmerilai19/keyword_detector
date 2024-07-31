# keyword_detector

TensorFlow ML based keyword detector using TensorFlow's *mini_speech_commands*

## Usage
Run the launch script:
```
launch.bat
```

or run manually:

1. Download data samples from TensorFlow's library
```
python download_dataset.py
```

2. Prepare a data.json file from the samples
```
python prepare_dataset.py
```

3. Train the network
```
python train.py
```

4. Create and test the end product
```
python keyword_detector.py
```
