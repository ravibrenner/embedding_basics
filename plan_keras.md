# Project Plan: Goodreads 10k Collaborative Filtering with Keras

## Phase 1: Data Acquisition and Wrangling (Pandas)
*   **Load and Inspect:** Load `ratings.csv` and `books.csv`. 
*   **Merge Metadata:** Join book metadata to the ratings dataframe.
*   **Clean the IDs:** Map raw `book_id` and `user_id` to contiguous integers (0 to N-1) using Pandas categorical codes. Save the mapping dictionaries.

## Phase 2: Building the Data Pipeline (TensorFlow)
*   **Train/Validation/Test Split:** Split the Pandas dataframe 80/10/10.
*   **Create the Dataset (`tf.data`):**
    *   While Keras can accept Pandas dataframes directly, using `tf.data.Dataset.from_tensor_slices()` is a better practice for performance and scalability.
    *   Construct the dataset to yield a tuple of inputs and targets: `({"user_input": users, "book_input": books}, ratings)`.
    *   Apply `.batch(1024).cache().prefetch(tf.data.AUTOTUNE)` to optimize data loading speed.

## Phase 3: Defining the Matrix Factorization Model (Keras Functional API)
*   **Input Layers:** Create two `tf.keras.layers.Input` layers, one for `user_input` and one for `book_input`.
*   **Embedding Layers:** 
    *   Create `tf.keras.layers.Embedding` layers for users and books. Apply `tf.keras.regularizers.l2(1e-5)` directly within the embedding layer definition to prevent overfitting.
    *   Create bias embedding layers (output dimension = 1) for users and books.
*   **Dot Product & Addition:** 
    *   Use `tf.keras.layers.Dot(axes=1)` to compute the interaction between the user and book embeddings.
    *   Use `tf.keras.layers.Add()` to add the dot product and the two bias terms together.
*   **Model Compilation:** 
    *   Define the model: `model = tf.keras.Model(inputs=[user_input, book_input], outputs=prediction)`.

## Phase 4: Training (The Keras Way)
*   **Compilation:** `model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')`.
*   **Callbacks:**
    *   Use `tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)` to automatically handle overfitting.
*   **Execution:** Call `model.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[early_stopping])`. This single line replaces the entire custom loop required in PyTorch.

## Phase 5: Extraction, Visualization, and Analysis
*   **Extract Weights:** Access the weights directly from the layer: `book_embeddings = model.get_layer('book_embedding').get_weights()[0]`.
*   **Dimensionality Reduction:** Use Scikit-learn's `TSNE` or `UMAP` to map the arrays to 2D.
*   **Visualization:** Create an interactive scatter plot with Plotly.
*   **Vector Math:** Use Scikit-learn's `cosine_similarity` module to find the nearest neighbors for a target book embedding.
