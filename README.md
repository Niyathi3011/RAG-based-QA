# Question-Answer System README

This repository contains code for a question-answering system. The system utilizes various libraries and models to process and generate answers to user queries. This README provides an overview of the code and explanations for the included code snippets.

## Code Explanations

1. `preprocess_text`:
   - This script contains code for text preprocessing. It uses the NLTK library to tokenize the text, remove punctuation and stopwords, and perform other preprocessing tasks.
 
   - This script demonstrates the usage of the question-answering system. It includes code for loading the necessary models and data, preprocessing the input, and generating answers based on the user's query.

3. `classify_passages`:
   - This script contains a function that classifies passages as positive or negative based on a relevance threshold. It uses the BM25 algorithm to calculate similarity scores between the query and passages, and assigns passages with scores above the threshold as positive.

4. `create_dataset`:
   - This script creates a dataset for training a model. It randomly selects negative passages to create positive-negative pairs and stores them in a dictionary format.

5. `train_model`:
   - This script trains a model using the created dataset. It defines hyperparameters such as batch size, learning rate, and loss function. It also includes steps for indexing and encoding passages, performing backpropagation and optimization, and saving the trained model.

6. `generate_embeddings`:
   - This script generates embeddings for a set of passages. It uses the DPR context encoder and tokenizer to encode the passages and extract their embeddings. The embeddings are then stored for further use.

7. `search_index`:
   - This script performs similarity search using the FAISS library. It creates an index using the HNSWFlat algorithm and adds the embeddings of passages to the index. It also demonstrates how to perform a search using a query embedding.

8. `generate_answers`:
   - This script generates answers to user queries using the T5 model. It combines the question and relevant context, encodes them, generates an answer using the model, and prints the generated answers. 
   - RAG Generator model from the hugging face has also been tried, but, the generated answers were not relevant or good enough. 
   - T5 performed better than the RAG Generator 

Please refer to the individual code files for more detailed explanations and instructions on running the code.

## Experiments : 

1. `classify_passages` : 
    - I have used cosine similarity to generate positive and negative passages, with the threshold for similarity > 0.55 for the contrastive learning. 
    - When BM25 was used for classifying positive and negative passages and then used for contrastive learning, the T5 model seemed to generate relevant answers.  

2. `models for generation` : 
    - RAG Generator, BART and T5 model have been used for the text generation. I had to stick to these models owing to the resource limitations on my google collab.  
    - T5 model, with the passages retrieved, performed better when compared to the other two models. 
    - These results can be seen in the respective notebooks, in the git repository. 

##Questions and Generated Answers : 

## Thoughts : 

1. `Dataset` : 
    - I have not generated any question answer pairs from the given pdfs. So, I had to stick to Retriever based augmented generation. 
    - Contrastive learning, proved to be helpful to retrieve the relevant passages with high similarity. 
    - But, there are couple of issues with the contrastive learning, as it is highly hyperparameter sensitive, and the boundary to generate       negative passages is questionable. 


