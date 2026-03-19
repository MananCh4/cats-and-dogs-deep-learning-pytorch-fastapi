# Cats vs Dogs Classifier (PyTorch + FastAPI)

An end-to-end deep learning project that classifies images of cats and dogs. It features a custom PyTorch neural network (`SimpleNN`) and a RESTful API built with FastAPI for real-time image classification.

## 🛠️ Tech Stack
* **Machine Learning:** PyTorch, Torchvision, Scikit-learn
* **Backend & API:** FastAPI, Uvicorn
* **Data Processing:** Pandas, NumPy, Pillow (PIL)

## 📂 Repository Structure

```text
📁 cats-vs-dogs-classifier-pytorch-fastapi
 ├── 📄 main.py                   # FastAPI application and predict endpoint
 ├── 📄 model_def.py              # PyTorch Neural Network architecture
 ├── 📄 cats_vs_dogs_model.pth    # Pre-trained model weights
 ├── 📄 cats_vs_dogs(train).csv   # Pre-processed training data (Flattened images)
 ├── 📄 cats_vs_dogs(test).csv    # Pre-processed testing data (Flattened images)
 ├── 📄 train_code.ipynb          # Data processing and model training notebook
 ├── 📄 test_code.ipynb           # Model evaluation notebook
 ├── 📄 predict_code.ipynb        # Local prediction testing notebook
 ├── 📄 requirements.txt          # Python dependencies
 └── 📄 Report.pdf                # Detailed project documentation

📌 Note on Dataset Handling
To keep this repository lightweight, the raw .jpg images are not included. Instead, pre-processed pixel data is provided directly in the .csv files.

The Jupyter Notebooks are designed to be flexible:

By default, the code will load the provided .csv files so you can run the training and testing cells immediately.

Want to process the raw images yourself? Simply create train_set/ and test_set/ folders in the root directory. Inside each, add cats/ and dogs/ subfolders containing your images. The notebooks will automatically detect the folders, process the images, and generate fresh CSVs for you.

⚙️ Setup and Installation 
To get the project running locally, simply clone the repository, activate a Python virtual environment, and run pip install -r requirements.txt to install all necessary dependencies such as FastAPI, PyTorch, and Pandas. Note on PyTorch Installation: The default requirements file installs the CPU version of PyTorch for quick and easy API testing; however, if you plan to re-train the neural network yourself, it is highly recommended to manually install the CUDA-enabled version from the official PyTorch website to take advantage of GPU acceleration.
