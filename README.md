# Continual Learning for Image Classification

A PyTorch implementation of a continual learning system designed to classify images from sequential data streams, featuring strategies to combat catastrophic forgetting and adapt to distribution shifts. This project was developed as part of the CS771 (Introduction to Machine Learning) course.



## About The Project

This project simulates a real-world machine learning scenario where data arrives sequentially and cannot be stored indefinitely. The core challenge is to train a model that can learn from new tasks without catastrophically forgetting the knowledge gained from previous tasks.

The system is designed to:
1.  Train an initial classifier on a small, labeled dataset.
2.  Incrementally learn from a stream of subsequent **unlabeled** datasets.
3.  Adapt to data from both stable (Task 1) and shifting (Task 2) distributions.

A custom prototype-based classifier is used, which updates its knowledge using pseudo-labels generated for the new, unlabeled data. To combat forgetting, a **rehearsal buffer** is implemented to store and replay key examples (exemplars) from past tasks during updates.

### Key Features
* **Sequential Learning:** Incrementally trains on 20 sequential datasets.
* **Pseudo-Labeling:** Learns from unlabeled data by generating its own labels.
* **Prototype-Based Classifier:** Uses a custom classifier based on class feature means (prototypes).
* **Catastrophic Forgetting Mitigation:** Implements a rehearsal buffer with exemplars to retain past knowledge.
* **Distribution Shift Analysis:** Evaluates model performance when the input data distribution changes over time.

### Built With
* **Python**
* **PyTorch**
* **NumPy**
* **EfficientNet** (as a pre-trained feature extractor)

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites
* Python 3.8+
* PyTorch, Torchvision, NumPy, Pillow
    ```sh
    pip install torch torchvision numpy Pillow
    ```

### Installation & Setup
1.  Clone the repo:
    ```sh
    git clone [https://github.com/your_username/your_repository.git](https://github.com/your_username/your_repository.git)
    ```
2.  Download the dataset from the project URL.
3.  Unzip the `dataset.zip` file and place the `dataset` folder in the root directory of the project.

### How to Run
The project is divided into two main parts, implemented as Jupyter notebooks:
1.  Run the **Task 1 Notebook** to train the model on the first 10 datasets and save the final state (`f10`).
2.  Run the **Task 2 Notebook** to load the state from Task 1 and continue training on the final 10 datasets.

## Results

The final model successfully learned from all 20 datasets. The rehearsal buffer strategy proved effective in reducing catastrophic forgetting compared to a naive sequential updating approach. The final accuracy matrix for Task 2 (models `f11`-`f20` evaluated on all 20 held-out datasets) is shown below.

The results highlight the model's stability on past tasks (due to rehearsal) but also show the significant challenge posed by the distribution shift in the later datasets (e.g., low accuracy on `D12`, `D19`).
## Future Improvements
* **Confidence-Based Learning:** Implement a threshold to only use high-confidence pseudo-labels for updates, which could improve performance on distribution-shifted data.
* **Dynamic Learning Rate (`alpha`):** Adjust the prototype update rate based on detected distribution shifts.
* **Advanced Rehearsal:** Explore more sophisticated strategies for selecting and managing exemplars in the buffer.

