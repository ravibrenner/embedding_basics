# Embedding Basics: Book Recommenders in Keras & PyTorch

A comparative implementation of collaborative filtering and matrix factorization using the Goodreads 10k dataset. This project explores the foundational mechanics of recommender systems, focusing on how embeddings are learned and utilized across different deep learning frameworks.

This was mainly a project for *me* to learn about these subjects. I relied quite a bit on reading documentation and using Gemini models to assist me along the way with planning and explanations. If you're looking for a more complete treatment of these topics, I suggest looking elsewhere.

## Project Overview
*   **Goal:** Predict user-book ratings (1–5) and extract meaningful vector representations (embeddings) for books.
*   **Dataset:** Goodreads 10k (User IDs, Book IDs, and Ratings).
*   **Task:** Explicit Recommendation (Regression) via Matrix Factorization.

## Key Learning Objectives
*   **Collaborative Filtering:** Understanding the "User-Item Interaction" matrix.
*   **Embedding Mechanics:** See how meaning is embedded into the matrix without explicit metadata.
*   **ML Frameworks:** Implementing the exact same math in Keras and PyTorch to observe architectural and workflow differences.

## Framework Comparison
A core part of this exercise was comparing Keras and PyTorch implementations of the same workflow. Here are some of my takeaways:

*   **Keras:** Much easier to write the code, especially for data loading, model training, and model testing. I would say the model definition felt easy, but not quite as well-packaged as PyTorch.
*   **PyTorch:** Harder to write the code, especially for data loading and the training and testing loops. I know there are ways to speed this up (i.e. PyTorch lightning), but I appreciate that having a lot of direct control is valuable. While I am not super well versed in class-based/object oriented workflows, I am starting to see why people like them. The whole model definition is right there with the model.

The two frameworks deal with the target variable very differently, which was interesting to learn about. In Keras it is mainly handled via the dataset creation step, where the target variable is the second part of the tuple. In PyTorch it is explicitly passed when calculating the loss in the training loop.

From reading online, I know that PyTorch is much more popular in recent years. Now that Keras/Tensorflow can use torch or jax as a backend, I wonder if that will change. That said, I see the benefits of the more verbose PyTorch approach. Once you get past the learning curve, the flexibility is huge, and the class-based programming is very clean.

## Architecture & Implementation
*   **The Math:** $\text{Score} = (\text{User} \cdot \text{Book}) + \text{User Bias} + \text{Book Bias}$.
*   **Regularization:** Using L2 weight decay to prevent the model from simply "memorizing" specific users.
*   **Optimization:** Adam optimizer with Mean Squared Error (MSE) loss.

## Analysis & Visualization
*   **Embedding Extraction:** Pulling weights from the trained model layers.
*   **Dimensionality Reduction:** Using **t-SNE** to project 32D book vectors into 2D space.

## Repository Structure
*   `books_analysis.ipynb`: The main exploration and prototyping playground.
*   `plan_keras.md` / `plan_pytorch.md`: Detailed implementation strategies for each framework (written by Gemini with my guidance).
*   `target_variable.md`: A deep dive into how ground truth labels are handled across frameworks (written by Gemini based on some questions I had).
*   `data/`: Source CSVs.
*   `output/`: Generated CSVs for learned user and book embeddings.
