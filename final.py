import numpy as np
import pandas as pd
import gradio as gr
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# Read data
try:
    data = pd.read_csv('data.csv')
except FileNotFoundError:
    raise FileNotFoundError("The file 'data.csv' was not found. Please make sure it is in the correct directory.")

# Validate data
if data.empty:
    raise ValueError("The dataset is empty. Please provide a valid dataset.")

# Map experience levels to descriptive labels
data['experience_level'] = data['experience_level'].replace({
    'SE': 'Senior',
    'EN': 'Entry level',
    'EX': 'Executive level',
    'MI': 'Mid level',
})

# Map employment types to descriptive labels
data['employment_type'] = data['employment_type'].replace({
    'FL': 'Freelancer',
    'CT': 'Contractor',
    'FT': 'Full-time',
    'PT': 'Part-time'
})

# Map company sizes to descriptive labels
data['company_size'] = data['company_size'].replace({
    'S': 'Small',
    'M': 'Medium',
    'L': 'Large',
})

# Map remote ratios to descriptive labels
data['remote_ratio'] = data['remote_ratio'].astype(str).replace({
    '0': 'On-Site',
    '50': 'Hybrid',
    '100': 'Remote',
})

# Group similar job titles into broader categories
def assign_broader_category(job_title):
    data_engineering = [
        "Data Engineer", "Data Analyst", "Analytics Engineer", "BI Data Analyst",
        "Business Data Analyst", "BI Developer", "BI Analyst",
        "Business Intelligence Engineer", "BI Data Engineer", "Power BI Developer"
    ]
    data_science = [
        "Data Scientist", "Applied Scientist", "Research Scientist",
        "3D Computer Vision Researcher", "Deep Learning Researcher",
        "AI/Computer Vision Engineer"
    ]
    machine_learning = [
        "Machine Learning Engineer", "ML Engineer", "Lead Machine Learning Engineer",
        "Principal Machine Learning Engineer"
    ]
    data_architecture = [
        "Data Architect", "Big Data Architect", "Cloud Data Architect",
        "Principal Data Architect"
    ]
    management = [
        "Data Science Manager", "Director of Data Science", "Head of Data Science",
        "Data Scientist Lead", "Head of Machine Learning", "Manager Data Management",
        "Data Analytics Manager"
    ]
    
    if job_title in data_engineering:
        return "Data Engineering"
    elif job_title in data_science:
        return "Data Science"
    elif job_title in machine_learning:
        return "Machine Learning"
    elif job_title in data_architecture:
        return "Data Architecture"
    elif job_title in management:
        return "Management"
    else:
        return "Other"

# Apply the function to create the 'job_category' column
data['job_category'] = data['job_title'].apply(assign_broader_category)

# Drop the original 'job_title' and 'salary_in_usd' columns if they exist
data = data.drop(columns=['job_title', 'salary_in_usd'], errors='ignore')

# Validate necessary columns
required_columns = ['experience_level', 'employment_type', 'job_category', 'salary']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"The following required columns are missing in the dataset: {missing_columns}")

# Identify categorical and numerical columns
categorical_columns = [
    'experience_level', 'employment_type', 'job_category',
    'salary_currency', 'employee_residence', 'company_location',
    'company_size', 'remote_ratio'
]

numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'salary' in numerical_columns:
    numerical_columns.remove('salary')  # Remove target variable

# Define target and features
target = 'salary'
features = categorical_columns + numerical_columns
X = data[features]
y = data[target]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[ 
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Apply transformations
X_processed = preprocessor.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Train RandomForest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Function to make predictions with the trained model
def predict_salary(experience_level, employment_type, job_category,
                   salary_currency, company_size, remote_ratio):
    # Set default values for the hidden fields
    work_year = '2023'
    employee_residence = 'US'
    company_location = 'US'
    
    # Create a dictionary for the new input data
    new_input_data = {
        'work_year': work_year,
        'experience_level': experience_level,
        'employment_type': employment_type,
        'job_category': job_category,
        'salary_currency': salary_currency,
        'employee_residence': employee_residence,
        'company_location': company_location,
        'company_size': company_size,
        'remote_ratio': remote_ratio
    }
    
    # Convert the new input data into the same format as the training data
    new_input_df = pd.DataFrame([new_input_data])

    # Apply the same preprocessing transformations
    new_input_processed = preprocessor.transform(new_input_df)

    # Predict the salary
    predicted_salary = model.predict(new_input_processed)

    return f"Predicted Salary: ${predicted_salary[0]:,.2f}"

# Function to create data analysis charts
def plot_charts():
    if data.empty:
        raise ValueError("The dataset is empty. Please provide valid data.")

    # Employment Type Distribution
    type_grouped = data['employment_type'].value_counts()
    fig1 = px.bar(type_grouped, x=type_grouped.index, y=type_grouped.values,
                  title="Employment Type Distribution", template='plotly_dark')

    # Salary by Work Year
    salary_by_year = data.groupby('work_year')['salary'].mean().reset_index()
    fig2 = px.line(salary_by_year, x='work_year', y='salary', 
                   title="Salary by Work Year", markers=True, template='plotly_dark')

    # Top 10 Job Categories
    top_job_categories = data['job_category'].value_counts().head(10)
    fig3 = px.bar(top_job_categories, x=top_job_categories.index, y=top_job_categories.values,
                  title="Top 10 Job Categories", template='plotly_dark')

    return fig1, fig2, fig3

# Create Gradio dashboard
def create_dashboard():
    # Tab 1: Salary Prediction
    inputs = [
        gr.Dropdown(choices=['Entry level', 'Mid level', 'Senior', 'Executive level'], label="Experience Level"),
        gr.Dropdown(choices=['Freelancer', 'Contractor', 'Full-time', 'Part-time'], label="Employment Type"),
        gr.Dropdown(choices=['Data Science', 'Data Engineering', 'Machine Learning', 'Data Architecture', 'Management'], label="Job Category"),
        gr.Textbox(label="Salary Currency"),
        gr.Dropdown(choices=['Small', 'Medium', 'Large'], label="Company Size"),
        gr.Dropdown(choices=['On-Site', 'Hybrid', 'Remote'], label="Remote Ratio")
    ]
    output = gr.Textbox(label="Predicted Salary")
    salary_interface = gr.Interface(
        fn=predict_salary, 
        inputs=inputs, 
        outputs=output, 
        live=True, 
        title="Salary Prediction",
        description="Predict salary based on job attributes."
    )
    
    # Tab 2: Data Analysis Charts
    chart_output1 = gr.Plot()
    chart_output2 = gr.Plot()
    chart_output3 = gr.Plot()
    data_analysis_interface = gr.Interface(
        fn=plot_charts,
        inputs=[],  # No inputs needed for this page
        outputs=[chart_output1, chart_output2, chart_output3],
        title="Data Analysis",
        description="Analyze Employment Types, Salary by Work Year, and Top 10 Job Categories."
    )
    
    # Combine both interfaces into one dashboard
    dashboard = gr.TabbedInterface(
        [salary_interface, data_analysis_interface], 
        tab_names=["Salary Prediction", "Data Analysis"]
    )
    return dashboard

# Launch the dashboard
dashboard = create_dashboard()
dashboard.launch()
