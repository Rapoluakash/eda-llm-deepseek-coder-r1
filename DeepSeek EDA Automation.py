import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama

# Main EDA function
def eda_analysis(file_path):
    df = pd.read_csv(file_path)
    
    # Handle missing values (safe assignment)
    for col in df.select_dtypes(include='number'):
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Summary & Missing Values
    summary = df.describe(include='all').to_string()
    missing = df.isnull().sum().to_string()
    
    # AI Insights from DeepSeek R1
    insights = generate_ai_insights(summary)
    
    # Visuals
    plots = generate_visualizations(df)

    return f"""✅ **EDA Complete**

📊 **Summary**
{summary}

❓ **Missing Values**
{missing}

🤖 **DeepSeek AI Insights**
{insights}
""", plots

# DeepSeek R1 chat
def generate_ai_insights(summary_text):
    prompt = f"You're a data analyst. Provide insights based on this data summary:\n\n{summary_text}"
    response = ollama.chat(
        model="deepseek-coder:latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# Create visualizations
def generate_visualizations(df):
    paths = []
    
    for col in df.select_dtypes(include='number'):
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, color="lightgreen", bins=30)
        plt.title(f"Distribution: {col}")
        path = f"{col}_hist.png"
        plt.savefig(path)
        plt.close()
        paths.append(path)

    if not df.select_dtypes(include='number').empty:
        plt.figure(figsize=(8, 5))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        heatmap_path = "heatmap.png"
        plt.savefig(heatmap_path)
        plt.close()
        paths.append(heatmap_path)
    
    return paths

# Gradio UI
demo = gr.Interface(
    fn=eda_analysis,
    inputs=gr.File(type="filepath", label="📁 Upload CSV"),
    outputs=[
        gr.Textbox(label="📋 EDA Report"),
        gr.Gallery(label="📊 Visuals")
    ],
    title="🔍 EDA Automation with DeepSeek R1",
    description="Upload your CSV dataset. This app will perform automatic EDA and explain it using DeepSeek-Coder R1 model."
)

demo.launch(share=True)
