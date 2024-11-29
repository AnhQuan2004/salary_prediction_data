import gradio as gr
import plotly.express as px
import pandas as pd

# Giả sử df là DataFrame đã được xử lý trước đó (bạn cần thay thế df bằng DataFrame thực tế của mình)
# Ví dụ về cách đọc dữ liệu từ file CSV
df = pd.read_csv('data.csv')  # Thay bằng đường dẫn đúng đến file dữ liệu của bạn

# Biểu đồ phân phối Employment Type
def plot_charts():
    # Employment Type Distribution
    type_grouped = df['employment_type'].value_counts()
    e_type = ['Full-Time', 'Part-Time', 'Contract', 'Freelance']
    fig = px.bar(x=e_type, y=type_grouped.values, 
                 color=type_grouped.index, 
                 color_discrete_sequence=px.colors.sequential.PuBuGn,
                 template='plotly_dark',
                 text=type_grouped.values, title='2.1.3. Employment Type Distribution')
    fig.update_layout(
        xaxis_title="Employment Type",
        yaxis_title="count",
        font=dict(size=17, family="Franklin Gothic")
    )
    fig.update_traces(width=0.5)

    # Salary by Work Year
    salary_by_year = df.groupby('work_year')['salary'].mean().reset_index()
    fig2 = px.line(salary_by_year, x='work_year', y='salary', 
                   title="Salary by Work Year",
                   markers=True, template='plotly_dark')
    fig2.update_layout(
        xaxis_title="Work Year",
        yaxis_title="Average Salary",
        font=dict(size=17, family="Franklin Gothic")
    )
    
    # Top 10 Job Titles
    top_job_titles = df['job_title'].value_counts().head(10)
    fig3 = px.bar(top_job_titles, x=top_job_titles.index, y=top_job_titles.values,
                  title="Top 10 Job Titles", template='plotly_dark')
    fig3.update_layout(
        xaxis_title="Job Title",
        yaxis_title="Count",
        font=dict(size=17, family="Franklin Gothic")
    )

    return fig, fig2, fig3

# Trang thứ hai sẽ gọi hàm `plot_charts` để tạo các biểu đồ.
def create_chart_interface():
    chart_output1 = gr.Plot()
    chart_output2 = gr.Plot()
    chart_output3 = gr.Plot()

    return gr.Interface(
        fn=plot_charts,
        inputs=[],
        outputs=[chart_output1, chart_output2, chart_output3],
        title="Data Analysis",
        description="Analyze Employment Types, Salary by Work Year, and Top 10 Job Titles."
    )

# Trang đầu tiên: Dự đoán mức lương
def predict_salary(work_year, experience_level, employment_type, job_category,
                   salary_currency, employee_residence, company_location,
                   company_size, remote_ratio):
    # Dự đoán lương (thực hiện như đã đề cập trước đó)
    # Giả sử bạn có một mô hình đã huấn luyện trước đó để thực hiện dự đoán
    # Phần này có thể được thay bằng mô hình đã được huấn luyện như RandomForestRegressor hoặc bất kỳ mô hình nào bạn đã dùng.
    predicted_salary = 100000  # Dự đoán giả sử (bạn có thể thay thế bằng mô hình thực tế)
    return f"Predicted Salary: ${predicted_salary}"

# Giao diện đầu tiên cho dự đoán mức lương
inputs = [
    gr.Number(label="Work Year", value=2023),
    gr.Dropdown(choices=['Entry level', 'Mid level', 'Senior', 'Executive level'], label="Experience Level"),
    gr.Dropdown(choices=['Freelancer', 'Contractor', 'Full-time', 'Part-time'], label="Employment Type"),
    gr.Dropdown(choices=['Data Science', 'Data Engineering', 'Machine Learning', 'Data Architecture', 'Management'], label="Job Category"),
    gr.Textbox(label="Salary Currency"),
    gr.Textbox(label="Employee Residence"),
    gr.Textbox(label="Company Location"),
    gr.Dropdown(choices=['Small', 'Medium', 'Large'], label="Company Size"),
    gr.Dropdown(choices=['On-Site', 'Hybrid', 'Remote'], label="Remote Ratio")
]

output = gr.Textbox(label="Predicted Salary")

# Giao diện cho phân tích dữ liệu
chart_interface = create_chart_interface()

# Tạo giao diện với Tabs
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Salary Prediction"):
            gr.Interface(fn=predict_salary, inputs=inputs, outputs=output).render()
        with gr.TabItem("Data Analysis"):
            chart_interface.render()

demo.launch()

