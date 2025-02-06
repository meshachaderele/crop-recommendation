

# Crop Recommendation Web App

This web application is built using FastAPI and provides crop recommendations based on various environmental inputs such as NPK levels, temperature, humidity, and rainfall. It leverages a pre-trained machine learning model for predictions, and it can be deployed locally or in a cloud environment.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.7 or higher
- `pip` or `conda` package manager

## Installation

### Step 1: Clone the Repository

First, clone the repository to your local machine:

```bash
git https://github.com/meshachaderele/crop-recommendation.git
cd crop-recommendation
```

### Step 2: Install Dependencies

Ensure that all the required dependencies are installed. The project includes a `requirements.txt` file that lists all necessary packages.

#### Using pip:

1. Create a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   - For **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - For **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. Install the dependencies from the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

#### Using Conda:

If you are using Conda, create and activate a Conda environment:

```bash
conda create --name crop_recommendation python=3.11
conda activate crop_recommendation
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Configure the Application

Ensure you have your configuration file (`config.yaml`) set up correctly. This file contains settings like the model file path and other configuration parameters needed for the app to work.

If you do not have the `config.yaml`, you can create it by copying the sample template (if available) and adjusting the values.

### Step 4: Run the App

Once everything is set up, you can start the application. Navigate to the `app` folder, and use **uvicorn** to run the FastAPI application.

Run the following command in your terminal:

```bash
uvicorn app:app --reload
```

This will start the development server with hot reloading enabled.

- The app will be accessible at `http://127.0.0.1:8000/` in your browser.
- The `--reload` flag ensures that the server automatically reloads whenever you make changes to the code.

### Step 5: Access the Web App

1. Open your browser and navigate to `http://127.0.0.1:8000/`.
2. Fill out the form with the necessary inputs (such as NPK levels, temperature, etc.).
3. Submit the form, and the app will display the recommended crop along with an associated image.

## Features

- **Input Form**: A simple HTML form allows users to input various environmental parameters.
- **Machine Learning Model**: The app uses a pre-trained machine learning model to predict the most suitable crop.
- **Image Display**: The recommended crop is shown along with an associated image.


