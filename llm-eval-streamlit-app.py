
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from QAMetrics import QAMetrics 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to create gauge plots in subplots
def plot_gauge_subplots(metrics, avg_scores):
    num_metrics = len(metrics)
    cols = 3
    rows = (num_metrics + cols - 1) // cols  # Calculate number of rows needed, rounding up

    # Create subplots with specs for indicator type
    fig = make_subplots(
        rows=rows, cols=cols, 
        subplot_titles=[f"Average {metric}" for metric in metrics],
        specs=[[{"type": "indicator"} for _ in range(cols)] for _ in range(rows)]
    )

    for i, (metric, avg_score) in enumerate(zip(metrics, avg_scores)):
        row = i // cols + 1
        col = i % cols + 1

        # Add trace only if the row and col are within the grid
        if row <= rows and col <= cols:
            # Determine the range and color scheme based on the metric
            if metric == "F1 Score":
                gauge_config = {
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#5C6BC0"},
                    'steps': [
                        {'range': [0, 0.4], 'color': "#F48FB1"},
                        {'range': [0.4, 0.75], 'color': "#FFAB91"},
                        {'range': [0.75, 1], 'color': "#A5D6A7"}
                    ],
                }
            else:
                gauge_config = {
                    'axis': {'range': [1, 5]},
                    'bar': {'color': "#5C6BC0"},
                    'steps': [
                        {'range': [1, 3], 'color': "#F48FB1"},
                        {'range': [3, 4], 'color': "#FFAB91"},
                        {'range': [4, 5], 'color': "#A5D6A7"}
                    ],
                }

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=avg_score,
                    number={'font': {'size': 20}}, 
                    gauge=gauge_config
                ),
                row=row, col=col
            )

    # Update layout to ensure all subplots have the same size and spacing
    fig.update_layout(
        height=250 * rows, 
        width=800, 
        annotations=[dict(
            x=annotation['x'],
            y=annotation['y'] + 0.1,  # Increase space by moving the title upwards
            text=annotation['text'],
            showarrow=False,
            xanchor='center',
            yanchor='bottom'
        ) for annotation in fig.layout.annotations]  # Adjusting individual subplot titles
    )

    st.plotly_chart(fig)

def plot_histograms_fixed_size(metrics, results_df):
    num_metrics = len(metrics)
    cols = 2  # Maximum 2 plots per row
    rows = (num_metrics + 1) // cols  # Calculate number of rows needed

    # Fixed size per subplot
    fixed_width = 6  # Width of each subplot in inches
    fixed_height = 4  # Height of each subplot in inches

    # Adjust size if there is only one metric
    if num_metrics == 1:
        fig, ax = plt.subplots(figsize=(fixed_width, fixed_height))
        axes = [ax]
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(fixed_width * cols, fixed_height * rows))
        axes = axes.flatten()  # Flatten in case of multiple axes

    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        if metric in results_df.columns:  # Ensure the metric exists in the DataFrame
            if metric == "F1 Score":
                sns.lineplot(x=results_df.index, y=results_df[metric], ax=ax, marker="o")
                xmin, xmax = 0, len(results_df)-1
                ax.set_xlim(xmin, xmax)  # Set the x-axis range dynamically
                ax.set_xticks(range(int(xmin), int(xmax) + 1))  # Set x-axis ticks to display only integers within the range
            
            else:
                sns.histplot(results_df[metric], bins=5, binrange=(1, 5), kde=False, ax=ax)
                xmin, xmax = 0, 5
                ax.set_xlim(0.5, 5.5)  # Shift x-axis limits to center the ticks
                ax.set_xticks([1, 2, 3, 4, 5])  # Set x-axis ticks
            
            ax.set_title(f"{metric}")
            ax.set_xlabel("Score" if metric != "F1 Score" else "Question Index")
            ax.set_ylabel("Frequency" if metric != "F1 Score" else metric)
            ax.grid(color='grey', linestyle=':', linewidth=0.5)

    # Hide any unused subplots (if there are fewer metrics than spaces)
    if num_metrics > 1:
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)


def compute_and_display_metrics(metrics_to_compute, df, question_col, ground_truth_col, answer_col, context_col=None):
    results = {}

    # Compute selected metrics using QAMetrics methods
    if "f1_score" in metrics_to_compute:
        results["F1 Score"] = metrics_calculator.compute_f1_score(df[ground_truth_col], df[answer_col])
    if "coherence_score" in metrics_to_compute:
        results["Coherence Score"] = [metrics_calculator.calculate_coherence_score(q, a) for q, a in zip(df[question_col], df[answer_col])]
    if "similarity_score" in metrics_to_compute:
        results["Similarity Score"] = [metrics_calculator.calculate_similarity_score(q, gt, a) for q, gt, a in zip(df[question_col], df[ground_truth_col], df[answer_col])]
    if context_col is not None:
        if "groundedness_score" in metrics_to_compute:
            results["Groundedness Score"] = [metrics_calculator.calculate_groundedness_score(c, a) for c, a in zip(df[context_col], df[answer_col])]
        if "relevance_score" in metrics_to_compute:
            results["Relevance Score"] = [metrics_calculator.calculate_relevance_score(c, q, a) for c, q, a in zip(df[context_col], df[question_col], df[answer_col])]

    # Combine original dataset features with the computed results
    results_df = df.copy()  # Start with the original dataset columns
    for metric, values in results.items():
        results_df[metric] = values  # Add each metric as a new column

    metrics = list(results.keys())

    # Prepare data for gauge plots if all metrics are computed
    avg_scores = [results_df[metric].mean() for metric in metrics]
    plot_gauge_subplots(metrics, avg_scores)
    
    # Plot the histograms with fixed subplot sizes
    plot_histograms_fixed_size(metrics, results_df)

    st.dataframe(results_df)
    return results 


# Streamlit app setup
st.title("RAG Evaluation")

# Sidebar setup
with st.sidebar:
    # Upload the dataset
    uploaded_file = st.file_uploader("Upload your Q&A dataset (json format)", type=["json"])
    
    # API Key input
    api_key = st.text_input("OpenAI API key", type="password")

    # Model selection dropdown
    model_name = st.selectbox(
        "Select the model to use:",
        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
    )
    
    # Temperature input
    temperature = st.text_input("Temperature:", value="0")

# Initialize QAMetrics with API key and selected model
if uploaded_file is not None and api_key:
    metrics_calculator = QAMetrics(api_key=api_key, model_name=model_name, temperature=float(temperature))  

if uploaded_file is not None:
    # Read questions, groundtruth, answers from the file
    file_content = uploaded_file.read().decode("utf-8")
    data = json.loads(file_content)

    # Convert the data into a DataFrame for easier manipulation
    df = pd.DataFrame(data)
    features = df.columns.tolist()

    # Context or No Context selection
    eval_mode = st.radio("Select Evaluation Mode", ["with context", "with no context"])

    # Drop-down menus to select relevant features
    question_col = st.selectbox("Select the Question feature", features)
    ground_truth_col = st.selectbox("Select the Ground Truth feature", features)
    answer_col = st.selectbox("Select the Answer feature", features)
    context_col = None
    if eval_mode == "with context":
        context_col = st.selectbox("Select the Context feature", features)

    # Metric checkboxes
    metrics_to_compute = []

    if st.checkbox("Compute F1 Score"):
        metrics_to_compute.append("f1_score")
    if st.checkbox("Compute Coherence Score"):
        metrics_to_compute.append("coherence_score")
    if st.checkbox("Compute Similarity Score"):
        metrics_to_compute.append("similarity_score")
    if eval_mode == "with context":
        if st.checkbox("Compute Groundedness Score"):
            metrics_to_compute.append("groundedness_score")
        if st.checkbox("Compute Relevance Score"):
            metrics_to_compute.append("relevance_score")

    if st.button("Compute Metrics"):
        results = compute_and_display_metrics(metrics_to_compute, df, question_col, ground_truth_col, answer_col, context_col)
        # Generate and display the conclusion
        conclusion = metrics_calculator.generate_llm_conclusion(results)
        st.subheader("Conclusion and Interpretation")
        st.write(conclusion)