# FitReady

FitReady is a Streamlit application that provides personalized workout recommendations based on user input. The application uses machine learning models to predict the user's average, maximum, and resting heart rate, and to recommend exercises based on the target muscles and equipment specified by the user.

## Features

- Predicts user's average, maximum, and resting heart rate based on their metrics.
- Recommends exercises based on target muscles and equipment.
- Interactive user interface built with Streamlit.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/vincentmughal11/fitready.git
    cd fitready
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to access the application.

## Models

### UserWorkoutModel

This class is used to train a model that predicts the user's average, maximum, and resting heart rate based on their metrics.

### ExerciseRecommender

This class is used to recommend exercises based on user input. It preprocesses the data, trains a RandomForestClassifier model, and provides recommendations based on target muscles and equipment.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
