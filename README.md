🌸 Iris Species Predictor

An interactive web application that predicts Iris flower species using a pre-trained Artificial Neural Network (ANN) model.
Built with TensorFlow and deployed through Streamlit, the app provides accurate, real-time predictions through a clean and intuitive user interface.

🌟 Features

🔮 Real-time Iris species prediction

🧠 Pre-trained ANN model ensuring high accuracy

⚙️ Simple, interactive web interface

📏 Input validation for reliable predictions

⚡ Fast and efficient processing

🛠️ Technologies Used

Python 3.x

TensorFlow / Keras – Deep learning framework

Streamlit – Web app development

NumPy – Numerical computations

Pandas – Data manipulation

Scikit-learn – Data preprocessing

Seaborn & Matplotlib – Visualization (during training)

📊 Input Features

The model uses the following input features:

Sepal Length (cm)

Sepal Width (cm)

Petal Length (cm)

Petal Width (cm)

🚀 How to Use

Visit the hosted Streamlit App

Enter the flower measurements in the input fields

Click “🔮 Predict Species”

Instantly view the predicted species

🔧 Local Setup
1. Clone the repository
git clone https://github.com/Dhineshsakthivel2007/ANN-iris-.git
cd ANN-iris-

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run app.py

📁 Project Structure
ANN-iris-/
│
├── app.py             # Streamlit application
├── my_model.h5        # Trained ANN model
├── classes.npy        # Numpy array of class labels
├── Iris.csv           # Dataset used for training
└── requirements.txt   # Required dependencies

🧠 Model Overview

The ANN model consists of:

Input layer: 4 neurons (for each feature)

Hidden layers: Two layers with ReLU activation

Output layer: 3 neurons with Softmax activation

Loss Function: Categorical Crossentropy
Optimizer: Adam
Accuracy: ~96% on test data

This architecture provides a strong balance between performance and simplicity for small-scale datasets.

🏁 Conclusion

This project showcases how deep learning can effectively classify data using a simple Artificial Neural Network and demonstrates AI deployment through Streamlit.
It bridges the gap between machine learning models and real-world applications, making model interaction easy, visual, and engaging.