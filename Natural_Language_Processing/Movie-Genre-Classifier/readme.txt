Movie Genre Classifier
----------------------------------------------------------
(Jupyter Notebook is to be run sequentially)


Dataset: Movie Dataset with Movie Descriptions and Titles as independent features and Movie Genre as class

Goal: Classify to which genre/class the movie belongs according to the movie descriptions.

Approach 1: Using given dataset
--------------------------------

Tasks Performed:
- Pre-processing the data: Using langdetect, googletrans to detect and translate text, and other traditional text pre-processing.
- Indexing the Vocuabulary using textVectorization
- Embedding: Creating Embedding Matrix and Embedding Layer using pre-trained GLOVE Embeddings.
- Modelling: Implementing a Bi-Directional LSTM and some Dense Layers on top.
- Evaluation: Test Accurracy: 59.12%


Approach 2: Using Data Augmentation for every class (movie genre)
-----------------------------------------------------------------

Additional to Approach 1 - Data Augmentation
Tasks Performed:
- Handles Data Imbalance
- Method 1: Converted text to 16 different languages and re-translated the text back to English. This created variations in the text descriptions.
- Method 2: Created a Markov Chain using the entire training data. Generated text description samples from this chain.
- Generated the title for the augmented Movie Descriptions randomly.
- Further processing and modelling same as Approach 1
- Evaluation: Test Accuracy: 57.91%
- After augmenting the data for classes which had less samples, the class-wise accuracy increases.
