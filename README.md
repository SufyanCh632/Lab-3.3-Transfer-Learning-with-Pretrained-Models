# Lab-3.3-Transfer-Learning-with-Pretrained-Models
This README explains the concept of **Transfer Learning**, a powerful technique where a model developed for one task is reused as the starting point for a model on a second task. This allows us to achieve high accuracy on complex datasets like CIFAR-10 with very little training time.

---

# Lab 3.3: Transfer Learning with Pretrained Models

In previous labs, we trained models from scratch. In this lab, we leverage **ResNet-18**, which has already been trained on the **ImageNet** dataset (1.2 million images, 1,000 classes). Because the early layers of a CNN learn general features like edges and textures, we can "borrow" that knowledge for our own specific task.



## 🛠️ The Transfer Learning Workflow

### 1. Load a Pretrained Model
We load the ResNet-18 architecture with `pretrained=True`. This downloads the weights that have already been optimized to recognize a vast array of objects.

### 2. "Freeze" the Backbone
We iterate through all the model parameters and set `requires_grad = False`.
* **Why?** We don't want to ruin the sophisticated filters the model has already learned. We only want to train the very last part of the network.

### 3. Replace the "Head"
The original ResNet was built to classify 1,000 categories. CIFAR-10 only has 10. We replace the final fully connected layer (`fc`) with a new, untrained linear layer:
```python
resnet.fc = nn.Linear(resnet.fc.in_features, 10)
```

### 4. Fine-Tuning vs. Feature Extraction
In this lab, we perform **Feature Extraction**:
* The "Backbone" (convolutional layers) remains frozen.
* Only the "Head" (the new `fc` layer) is trained.
* **Result:** Training is incredibly fast because we are only optimizing a single linear layer instead of millions of parameters.

---

## 🏗️ Implementation Details

* **Data Preparation:** CIFAR-10 images are normalized to a mean/std of $0.5$ to match the expected input range of the pretrained model.
* **Optimizer:** We pass only `resnet.fc.parameters()` to the Adam optimizer. This ensures that even if we accidentally called `backward()`, the frozen weights wouldn't change.
* **Device:** The model and data are moved to **GPU (CUDA)** to ensure the forward passes are near-instantaneous.

---

## 📊 Summary of Advantages

| Feature | Training From Scratch | Transfer Learning |
| :--- | :--- | :--- |
| **Data Needed** | Large amounts (10k+) | Can work with very small datasets |
| **Time** | Hours/Days | Minutes |
| **Accuracy** | Starts from zero | Starts with "visual common sense" |
| **Compute** | High | Low (since most weights are frozen) |

---

## 🚀 How to Use
1.  Run the script to download the ResNet-18 weights and CIFAR-10.
2.  Observe the training loop: Notice how quickly the model reaches a high accuracy (often $>70\%$ after just one or two epochs).
3.  Check the final test accuracy to see the power of using a model that already "knows" what objects look like.

> **Key Takeaway:** Transfer Learning is the "cheat code" of Deep Learning. It is almost always better to start with a pretrained model than to initialize weights randomly, especially when working with image or text data.
