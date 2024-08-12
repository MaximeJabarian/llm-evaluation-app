# LLM Evaluation App

This repository contains a Streamlit application designed for evaluating the performance of Question-Answering (QA) models using various metrics. The app provides a user-friendly interface to upload datasets, select models, and visualize the results through intuitive graphs and plots.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Features](#features)
- [License](#license)

## Installation

To get started with the LLM Evaluation App, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/MaximeJabarian/llm-evaluation-app.git
   cd llm-evaluation-app
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**

   ```bash
   streamlit run llm-eval-streamlit-app.py
   ```

## Usage

1. **Upload your Dataset:**
   - On the sidebar, click the "Upload your Q&A dataset" button and select your JSON file. The dataset should be in the correct format, containing fields such as question, answer (model inference), ground_truth (expected answer), and optionally context.

2. **Enter API Key:**
   - Input your OpenAI API key to use the models for evaluation.

3. **Select the Model:**
   - Choose the model you want to evaluate from the dropdown menu (e.g., GPT-4o-mini, GPT-4o, GPT-4-turbo).

4. **Adjust Evaluation Settings:**
   - Choose whether to include context in the evaluation and select which metrics to compute (e.g., F1 Score, Coherence Score).

5. **Visualize Results:**
   - Once the metrics are computed, results are displayed through various graphs, including gauge plots and histograms, providing a comprehensive view of the model's performance.

## Files

- `QAMetrics.py`: Contains the class and methods for computing the evaluation metrics.
- `README.md`: The readme file you are currently reading.
- `demo_fast.gif`: A GIF demonstrating the app in action.
- `llm-eval-streamlit-app.py`: The main Streamlit application file.
- `requirements.txt`: A list of Python dependencies required to run the app.

## Features

- **Upload Q&A Dataset:** Easily upload datasets for evaluation.
- **Model Selection:** Choose from different models to evaluate their performance.
- **Metrics Computation:** Compute various metrics such as F1 Score, Coherence Score, Similarity Score, Groundedness Score, and Relevance Score.
- **User-Friendly Visualization:** Visualize the results using intuitive and interactive graphs.
- **Textual Interpretation:** Get a textual summary of the model's performance based on computed metrics.

## License

This project is open-source and available under the [MIT License](LICENSE).

