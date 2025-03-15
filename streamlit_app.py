import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure the page
st.set_page_config(page_title="TIMSS Dataset Viewer", layout="wide")

# Header
st.header('TIMSS 2023 Mathematics Dashboard')
st.write('This dashboard explores the TIMSS 2023 data for Irish 8th grade students, focusing on mathematics performance and factors influencing it.')

# Function to load data
def load_data(file_path):
    """Load data from CSV file if it exists, otherwise from SPSS"""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif os.path.exists(file_path.replace('.csv', '.sav')):
            import pyreadstat
            df, meta = pyreadstat.read_sav(file_path.replace('.csv', '.sav'))
            return df
        else:
            st.error(f"File not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Overview", "Student Factors", "Teacher Factors", "School Context"])

# Load data (checking for CSV first, then SPSS if needed)
if os.path.exists("../data/processed_student_data.csv"):
    student_data = load_data("../data/processed_student_data.csv")
else:
    # If not preprocessed, load and merge raw data
    st.sidebar.warning("Processed data not found, loading raw data...")
    df_ach = load_data("../orig/SPSS/bsairlm8.sav")
    df_bg = load_data("../orig/SPSS/bsgirlm8.sav")
    
    # Find common ID columns
    common_ids = list(set(df_ach.columns).intersection(set(df_bg.columns)))
    id_cols = [col for col in common_ids if col.startswith('ID')]
    
    # Merge data
    student_data = pd.merge(
        df_ach, 
        df_bg,
        on=id_cols,
        how="inner",
        suffixes=('_ach', '_bg')
    )
    
    # Create math score
    math_vars = [col for col in df_ach.columns if col.startswith('BSMMAT')]
    math_vars_merged = [f"{var}_ach" for var in math_vars]
    student_data['math_score'] = student_data[math_vars_merged].mean(axis=1)

# Similarly, load teacher and school data
if os.path.exists("../data/teacher_data.csv"):
    teacher_data = load_data("../data/teacher_data.csv")
else:
    teacher_data = load_data("../orig/SPSS/btmirlm8.sav")
    
if os.path.exists("../data/school_data.csv"):
    school_data = load_data("../data/school_data.csv")
else:
    school_data = load_data("../orig/SPSS/bcgirlm8.sav")

# Overview page
if page == "Overview":
    st.title("Mathematics Performance Overview")
    
    # Key statistics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Students", f"{len(student_data):,}")
    with col2:
        st.metric("Average Math Score", f"{student_data['math_score'].mean():.1f}")
    with col3:
        st.metric("Median Math Score", f"{student_data['math_score'].median():.1f}")
    
    # Score distribution histogram
    st.subheader("Distribution of Math Scores")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(student_data['math_score'], bins=30, kde=True, ax=ax)
    
    # Add TIMSS benchmark lines
    ax.axvline(x=400, color='r', linestyle='--', label='Low (400)')
    ax.axvline(x=475, color='orange', linestyle='--', label='Intermediate (475)')
    ax.axvline(x=550, color='g', linestyle='--', label='High (550)')
    ax.axvline(x=625, color='b', linestyle='--', label='Advanced (625)')
    
    ax.set_xlabel('Mathematics Score')
    ax.set_ylabel('Number of Students')
    ax.set_title('Distribution of Mathematics Scores with TIMSS Benchmarks')
    ax.legend()
    
    st.pyplot(fig)
    
    # Add explanation of benchmarks
    st.markdown("""
    **TIMSS Benchmarks:**
    - **Advanced (625+)**: Students can apply their understanding in complex situations
    - **High (550-624)**: Students can apply knowledge to solve problems
    - **Intermediate (475-549)**: Students can apply basic mathematical knowledge
    - **Low (400-474)**: Students have some basic mathematical knowledge
    """)

# Student Factors page
elif page == "Student Factors":
    st.title("Student Background and Attitudes")
    
    # Gender analysis
    st.subheader("Gender and Mathematics Performance")
    
    # Gender variable could be named differently
    gender_var = 'BSBG01' if 'BSBG01' in student_data.columns else 'ITSEX'
    
    if gender_var in student_data.columns:
        # Filter out NaN values
        gender_data = student_data.dropna(subset=[gender_var])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution
            fig, ax = plt.subplots(figsize=(6, 6))
            gender_counts = gender_data[gender_var].value_counts()
            ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
            ax.set_title('Gender Distribution')
            st.pyplot(fig)
            
        with col2:
            # Performance by gender
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=gender_var, y='math_score', data=gender_data, ax=ax)
            ax.set_title('Mathematics Performance by Gender')
            ax.set_xlabel('Gender')
            ax.set_ylabel('Mathematics Score')
            st.pyplot(fig)
        
        # Statistics table
        gender_stats = gender_data.groupby(gender_var)['math_score'].agg(['mean', 'median', 'std', 'count']).reset_index()
        gender_stats.columns = ['Gender', 'Mean Score', 'Median Score', 'Std Deviation', 'Count']
        st.dataframe(gender_stats)
    else:
        st.warning("Gender variable not found in the dataset")
    
    # Books at home analysis
    st.subheader("Home Educational Resources")
    books_var = 'BSBG04' if 'BSBG04' in student_data.columns else None
    
    if books_var and books_var in student_data.columns:
        books_data = student_data.dropna(subset=[books_var])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=books_var, y='math_score', data=books_data, ax=ax)
        ax.set_title('Mathematics Performance by Books at Home')
        ax.set_xlabel('Number of Books at Home')
        ax.set_ylabel('Mathematics Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Average scores by books
        books_avg = books_data.groupby(books_var)['math_score'].mean().reset_index()
        books_avg.columns = ['Books at Home', 'Average Score']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Books at Home', y='Average Score', data=books_avg, ax=ax)
        ax.set_title('Average Mathematics Score by Books at Home')
        ax.set_xlabel('Books at Home')
        ax.set_ylabel('Average Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Books at home variable not found in the dataset")
    
    # Student attitudes
    st.subheader("Student Attitudes Toward Mathematics")
    
    # Check for variables that might indicate attitudes
    attitude_vars = {
        'BSBGSLM': 'Student Likes Mathematics',
        'BSBGSCM': 'Student Confident in Mathematics'
    }
    
    for var, label in attitude_vars.items():
        if var in student_data.columns:
            attitude_data = student_data.dropna(subset=[var])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=var, y='math_score', data=attitude_data, ax=ax)
            ax.set_title(f'Mathematics Performance by {label}')
            ax.set_xlabel(label)
            ax.set_ylabel('Average Mathematics Score')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning(f"{label} variable not found in the dataset")

# Teacher Factors page
elif page == "Teacher Factors":
    st.title("Teacher and Instruction Factors")
    
    # Display teacher data overview
    st.subheader("Teacher Demographics")
    st.write(f"Number of teachers: {len(teacher_data)}")
    
    # Teacher experience
    teacher_exp_var = 'BTBG01' if 'BTBG01' in teacher_data.columns else None
    
    if teacher_exp_var and teacher_exp_var in teacher_data.columns:
        exp_counts = teacher_data[teacher_exp_var].value_counts().reset_index()
        exp_counts.columns = ['Years of Experience', 'Count']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Years of Experience', y='Count', data=exp_counts, ax=ax)
        ax.set_title('Distribution of Teaching Experience')
        ax.set_xlabel('Years of Experience')
        ax.set_ylabel('Number of Teachers')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Teacher experience variable not found in the dataset")
    
    # Teacher preparation
    st.subheader("Teacher Preparation and Professional Development")
    
    teacher_edu_var = 'BTBG04' if 'BTBG04' in teacher_data.columns else None
    
    if teacher_edu_var and teacher_edu_var in teacher_data.columns:
        edu_counts = teacher_data[teacher_edu_var].value_counts().reset_index()
        edu_counts.columns = ['Education Level', 'Count']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Education Level', y='Count', data=edu_counts, ax=ax)
        ax.set_title('Teacher Education Levels')
        ax.set_xlabel('Highest Education Level')
        ax.set_ylabel('Number of Teachers')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Teacher education variable not found in the dataset")

# School Context page
elif page == "School Context":
    st.title("School Context Factors")
    
    # Display school data overview
    st.subheader("School Demographics")
    st.write(f"Number of schools: {len(school_data)}")
    
    # School location
    school_loc_var = 'BCBG05A' if 'BCBG05A' in school_data.columns else None
    
    if school_loc_var and school_loc_var in school_data.columns:
        loc_counts = school_data[school_loc_var].value_counts().reset_index()
        loc_counts.columns = ['Location', 'Count']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Location', y='Count', data=loc_counts, ax=ax)
        ax.set_title('Distribution of Schools by Location')
        ax.set_xlabel('School Location')
        ax.set_ylabel('Number of Schools')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("School location variable not found in the dataset")
    
    # School composition
    st.subheader("School Composition and Climate")
    
    # Economic background of students
    econ_var = 'BCBG03A' if 'BCBG03A' in school_data.columns else None
    
    if econ_var and econ_var in school_data.columns:
        econ_counts = school_data[econ_var].value_counts().reset_index()
        econ_counts.columns = ['Economic Disadvantage', 'Count']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Economic Disadvantage', y='Count', data=econ_counts, ax=ax)
        ax.set_title('Schools by Percentage of Economically Disadvantaged Students')
        ax.set_xlabel('Percentage of Economically Disadvantaged Students')
        ax.set_ylabel('Number of Schools')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Economic disadvantage variable not found in the dataset")
    
    # School emphasis on academic success
    emphasis_var = 'BCBGEAS' if 'BCBGEAS' in school_data.columns else None
    
    if emphasis_var and emphasis_var in school_data.columns:
        emphasis_counts = school_data[emphasis_var].value_counts().reset_index()
        emphasis_counts.columns = ['Emphasis on Academic Success', 'Count']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Emphasis on Academic Success', y='Count', data=emphasis_counts, ax=ax)
        ax.set_title('School Emphasis on Academic Success')
        ax.set_xlabel('Level of Emphasis')
        ax.set_ylabel('Number of Schools')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("School emphasis variable not found in the dataset")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Dashboard created for Data Mining module")
st.sidebar.info("Data source: TIMSS 2023 Ireland Grade 8 dataset")

