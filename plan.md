# Project Plan: Goodreads 10k Collaborative Filtering with PyTorch

## Phase 1: Data Acquisition and Wrangling (Pandas)
*   **Load and Inspect:** Load `ratings.csv` and `books.csv`. 
*   **Merge Metadata:** Join book titles/authors to the ratings dataframe for interpretability.
*   **Clean the IDs:** Map raw `book_id` and `user_id` to contiguous integers (0 to N-1). 
    *   *Critical Review Note:* Always save these mapping dictionaries (e.g., using `pickle` or `json`). If you want to use this model later, you must know which index corresponds to which real-world book.

## Phase 2: Building the Data Pipeline (PyTorch)
*   **Train/Validation/Test Split:** 
    *   *Critical Review Improvement:* An 80/20 train/val split is basic. For a robust evaluation, use an 80/10/10 Train/Validation/Test split. You need an unseen test set to evaluate final generalization after tuning hyperparameters on the validation set.
*   **Create the Dataset:** Write a custom `torch.utils.data.Dataset` class that yields `(user_idx, book_idx, rating)`.
*   **Initialize DataLoaders:** Create `DataLoader` objects for train, validation, and test sets. Use a batch size of 1024 or 2048 (collaborative filtering can handle large batches).

## Phase 3: Defining the Matrix Factorization Model (PyTorch)
*   **The Architecture:** Subclass `nn.Module`. Include `nn.Embedding` for users, books, user biases, and book biases.
*   **The Forward Pass:** Compute the dot product of user and book embeddings, then add both biases.
*   **Initialization:** 
    *   *Critical Review Improvement:* Neural networks are sensitive to initialization. Explicitly initialize the embedding weights using a normal distribution with a small standard deviation (e.g., `std=0.01`) and initialize biases to zero.

## Phase 4: The Training and Validation Loop
*   **Loss Function:** Mean Squared Error (`nn.MSELoss()`).
*   **Optimizer:** Adam (`torch.optim.Adam`).
    *   *Critical Review Improvement (Regularization):* Collaborative filtering models are notorious for overfitting (memorizing the training users). You **must** add weight decay (L2 penalty) to your Adam optimizer (e.g., `weight_decay=1e-5`). 
*   **The Loop:** Write the manual training loop.
    *   After every training epoch, iterate through the Validation DataLoader (using `torch.no_grad()`) to compute the validation loss. 
    *   Implement early stopping: if validation loss increases for 3 consecutive epochs, stop training and revert to the best weights.

## Phase 5: Extraction, Visualization, and Analysis
*   **Extract Weights:** `book_embeddings = model.book_embeddings.weight.detach().cpu().numpy()`.
*   **Dimensionality Reduction:** Use `sklearn.manifold.TSNE` or `UMAP` (often superior to t-SNE for preserving global structure) to project the 32D embeddings into 2D.
*   **Visualization:** Plot the 2D points using Plotly for interactive hover-text.
*   **Vector Math:** Implement a function to calculate Cosine Similarity between a given book's vector and the rest of the matrix to build a "Nearest Neighbors" recommendation system.
