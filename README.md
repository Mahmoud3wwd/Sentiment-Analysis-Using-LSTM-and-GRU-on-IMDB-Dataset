# Sentiment Analysis Using LSTM and GRU on IMDB Dataset

## Project Overview
This project implements sentiment analysis on movie reviews from the IMDB dataset using two different deep learning architectures: Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU). The goal is to classify reviews as either positive or negative.

## Dataset
The dataset used in this project is the **IMDB Dataset** from kaggle in this link :https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews/input?select=IMDB+Dataset.csv
which contains movie reviews labeled as positive or negative. The dataset is available as a CSV file named `IMDB Dataset.csv`.

### Dataset Description
- **Columns**:
  - `review`: The text of the movie review.
  - `sentiment`: The sentiment label, either "positive" or "negative".
  
- **Total Samples**: [Insert Total Count]

## Methodology

### 1. Data Preprocessing
- **Loading Data**: The dataset is loaded using `pandas`.
- **Missing Values**: Checked and confirmed there are no missing values.
- **Data Splitting**: The dataset is split into training (80%) and testing (20%) sets using `train_test_split` from the `sklearn` library.

### 2. Text Processing
- **Tokenization**: Text reviews are converted into sequences of integers using the `Tokenizer` from Keras.
  - **Vocabulary Size**: Limited to the top 10,000 most frequent words.
  - **Out-of-Vocabulary Token**: Used "<OOV>" for unknown words.
- **Padding**: Sequences are padded to a maximum length of 120 to ensure uniform input size for the models.

### 3. Label Encoding
Sentiment labels are converted into binary values:
- "positive" → 1
- "negative" → 0

### 4. Model Development
#### A. LSTM Model
- **Architecture**:
  - **Embedding Layer**: Transforms words into dense vectors.
  - **LSTM Layer**: Captures sequential dependencies in the text.
  - **Dense Layer**: Fully connected layer with ReLU activation.
  - **Dropout Layer**: Regularization to prevent overfitting.
  - **Output Layer**: Sigmoid activation for binary classification.
  
- **Compilation**: The model is compiled using `binary_crossentropy` as the loss function, `adam` as the optimizer, and accuracy as the evaluation metric.

- **Training**: The model is trained for 10 epochs with early stopping based on validation loss.

#### B. GRU Model
- **Architecture**: Similar to the LSTM model, but utilizes a GRU layer instead of LSTM to capture dependencies in the text.
- **Compilation**: The same procedure as the LSTM model is followed.
- **Training**: The GRU model is trained for 10 epochs with early stopping.

### 5. Model Evaluation
Both models are evaluated on the test set, and performance metrics are calculated:
- **Accuracy**: Proportion of true results in total predictions.
- **Precision**: Proportion of true positive results in all positive predictions.
- **Recall**: Proportion of true positive results in all actual positives.
- **F1 Score**: Harmonic mean of precision and recall.

## Conclusion
This project demonstrates the effectiveness of LSTM and GRU models in classifying movie reviews from the IMDB dataset. Further enhancements can be made through hyperparameter tuning and exploring advanced architectures.

