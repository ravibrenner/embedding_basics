# Role of the 'Ratings' Target Variable

In both Keras and PyTorch implementations, the `ratings` variable (1–5 scale) is the **ground truth** the model aims to predict. This is an **explicit recommendation** (regression) task.

## 1. Data Loading (The "Mapping" Step)
The connection to the target variable begins during data preparation, where the ratings are designated as labels.

*   **Keras:** Uses `tf.data.Dataset.from_tensor_slices(({inputs}, labels))`. The `rating` column is passed as the second element of the tuple, identifying it as the target $y$.
*   **PyTorch:** The `BookRatingsDataset` class returns `(user_id, book_id, rating)` in its `__getitem__` method. The rating is bundled with the features for each batch.

## 2. Model Specification (The "Agnostic" Step)
The **Model Architecture** itself is agnostic to the ratings. Its only job is to perform the math:
$$\text{Output Score} = (\text{User Embedding} \cdot \text{Book Embedding}) + \text{User Bias} + \text{Book Bias}$$
The architecture defines *how* to generate a prediction, but it doesn't "know" what that prediction should represent until the training phase.

## 3. Training Loop (The "Objective" Step)
This is where the model is forced to align its output with the actual ratings via a **Loss Function** (Mean Squared Error).

*   **Keras:** `model.compile(loss='mse')` and `model.fit(train_ds)`. Keras internally takes the model's output and calculates the error against the labels provided in the dataset.
*   **PyTorch:** The connection is manual and explicit. In the loop, you fetch the `rating` from the batch and pass it directly into the loss function: `loss = criterion(model_output, rating)`.

## Summary: Architecture vs. Objective
| Phase | Responsibility | Target Variable Presence |
| :--- | :--- | :--- |
| **Architecture** | Math to generate a single "score" | **Absent** |
| **Data Loading** | Mapping features to labels | **Present** (as labels) |
| **Training/Loss** | Minimizing error between score and label | **Critical** (as the target) |

### Transition to Implicit Recommendation
To switch to **Implicit Recommendation** (tracking interactions instead of preferences), you would keep the **same architecture** but change the **target variable** to a binary $0/1$ (did the user interact?) and switch the loss function to Binary Crossentropy.
