
# NucShapeNet-DCDN

## Model Architecture: Dual-Stream CNN for Nucleosome Positioning Prediction 

This project implements a hybrid dual-stream convolutional neural network (CNN) for nucleosome positioning, as described in `model_npcdn.py`. The architecture consists of two parallel streams:

- **Stream 1: Dilated Conv2D**
	- Captures global sequence patterns using a series of dilated 2D convolutions with increasing dilation rates.
	- Input is reshaped to (145, 12, 1) and processed through several Conv2D layers with batch normalization, ReLU activation, and dropout.
	- Global average pooling is applied to summarize features.

- **Stream 2: DenseNet-121 Style Conv2D**
	- Focuses on local motif extraction using a DenseNet-inspired block structure.
	- Initial Conv2D and pooling, followed by multiple dense blocks (each with batch normalization, ReLU, Conv2D, dropout, and concatenation).
	- Transition layers compress and pool features between blocks.
	- Global average pooling is applied at the end.

The outputs of both streams are concatenated and passed through fully connected layers with dropout, ending in a sigmoid output for binary classification (nucleosomal vs linker).

---

## Usage

### 1. Data Encoding

Prepare your data by encoding FASTA files:

```
python Data_encoded.py -p <Fasta_file_Path> -f <Fasta_filename> -o <Output_file_Path>
```
- `-p <Fasta_file_Path>`: Path to the raw DNA sequence FASTA file
- `-f <Fasta_filename>`: Name of the FASTA file
- `-o <Output_file_Path>`: Output path for the encoded data

### 2. Training the Model

Train the Dual-Stream CNN with:

```
python training.py -p <Pickle_file_Path> -o <Output_file_Path> -e <Experiments_Name>
```
- `-p <Pickle_file_Path>`: Path to the encoded pickle file
- `-o <Output_file_Path>`: Output path for the trained model
- `-e <Experiments_Name>`: Name for the experiment/model folder

### 3. Prediction

Make predictions using a trained model:

```
python predict.py -p <Model_file_Path> -e <Experiments_Name>
```
- `-p <Model_file_Path>`: Path to the trained model
- `-e <Experiments_Name>`: Name of the experiment/model folder

---

## File Overview

- `model_npcdn.py`: Dual-Stream CNN model definition
- `Data_encoded.py`: Data encoding script
- `training.py`: Model training script
- `predict.py`: Prediction script

---
