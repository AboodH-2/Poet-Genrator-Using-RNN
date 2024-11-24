# Poet-Genrator-Using-RNN
 # Poet Generator using RNN

## Overview
This project is a text generation tool that creates poetic text using a Recurrent Neural Network (RNN) implemented with TensorFlow and Keras. The model is trained on a subset of Shakespeare's works and generates text based on a seed sequence.

## Features
- Utilizes LSTM layers for text generation.
- Customizable text generation based on temperature.
- Generates poetic or prose-like text.
- Model training and pre-trained model loading options.

## Requirements
- Python 3.7+
- TensorFlow 2.x
- Numpy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/poet-generator.git
   cd poet-generator
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) If you plan to use a GPU, ensure you have the correct CUDA and cuDNN versions installed. Refer to the [TensorFlow GPU support guide](https://www.tensorflow.org/install/gpu).

## Usage

### Training the Model
To train the model:
1. Open the `main.py` file and ensure the training block is uncommented:
   ```python
   # Prepare sequences and labels
   sentences = []
   next_characters = []

   for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
       sentences.append(text[i: i + SEQ_LENGTH])
       next_characters.append(text[i + SEQ_LENGTH])

   x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
   y = np.zeros((len(sentences), len(characters)), dtype=bool)

   for i, sentence in enumerate(sentences):
       for t, char in enumerate(sentence):
           x[i, t, char_to_index[char]] = 1
       y[i, char_to_index[next_characters[i]]] = 1

   model.fit(x, y, batch_size=256, epochs=4)
   model.save('textgenerator.model')
   ```
2. Run the script:
   ```bash
   python main.py
   ```

### Generating Text
To generate text using the pre-trained model:
1. Ensure the training block is commented out in `main.py`.
2. Run the script:
   ```bash
   python main.py
   ```
3. The script will output generated text at different temperatures (e.g., 0.2, 0.4, 0.6, 0.8, 1.0). Adjust the temperature parameter to control the creativity of the generated text.

### Sample Output
```text
----------0.2----------
shall i compare thee to a summer's day? thou art more lovely and more temperate:...

----------0.6----------
shall i compare thee to a summer's day? thou art more lovely and more temperate:...

----------1----------
shall i compare thee to a summer's day? thou art more lovely and more temperate:...
```

## File Structure
- `main.py`: Main script for training and text generation.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.

## Customization
- **Text Data**: Modify the `filepath` variable to train on a different text dataset.
- **Sequence Length**: Change the `SEQ_LENGTH` and `STEP_SIZE` values for different sequence slicing.
- **Model Architecture**: Update the LSTM layer parameters or add additional layers.

## Troubleshooting
- **GPU Warnings**: If you see warnings about missing GPU libraries, ensure you have the correct CUDA/cuDNN versions installed or ignore them to run on the CPU.
- **UnboundLocalError**: Ensure that variable names in the code are consistent and not overwritten.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [TensorFlow](https://www.tensorflow.org/) for the machine learning framework.
- [Shakespeare Dataset](https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt) for the training text.


