# ClassifiX - AI Image Classifier

An AI-powered image classification web application built using Streamlit and TensorFlow. Upload an image and the model predicts the object category using deep learning.

## Features

* Upload image files directly from browser
* AI-based image classification
* Fast predictions using TensorFlow
* Simple and responsive Streamlit UI
* Supports common image formats

## Tech Stack

* Python 3.11
* TensorFlow 2.15
* Streamlit
* NumPy
* Pillow

## Installation

Clone the repository:

```bash
git clone https://github.com/sajeerrr/AI-Image-Classifier.git
cd AI-Image-Classifier
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run main.py
```

## Project Structure

```text
AI-Image-Classifier/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```

## Deployment

This project can be deployed for free using:

* Streamlit Community Cloud

## Requirements

```text
streamlit
tensorflow==2.15.0
numpy<2.0
pillow
```

## Future Improvements

* Multiple model support
* Better UI design
* Real-time webcam classification
* Confidence score visualization
* Custom trained models

