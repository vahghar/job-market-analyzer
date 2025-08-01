import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd
import re
import numpy as np
from wordcloud import WordCloud

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load data
df = pd.read_csv("adzuna_all_jobs.csv", delimiter=",", quoting=3, encoding="utf-8", on_bad_lines='skip', engine='python')

# Enhanced skill categories with variations and synonyms
SKILL_CATEGORIES = {
    'Programming Languages': {
        'Python': ['python', 'py', 'python3'],
        'R': [r'\br\b', 'r programming', 'r language'],
        'SQL': ['sql', 'mysql', 'postgresql', 'sqlite', 'plsql', 't-sql'],
        'Java': ['java', 'java8', 'java 8'],
        'JavaScript': ['javascript', 'js', 'node.js', 'nodejs'],
        'Scala': ['scala'],
        'SAS': ['sas programming', 'sas software']
    },
    'Analytics & BI Tools': {
        'Excel': ['excel', 'microsoft excel', 'advanced excel', 'pivot tables'],
        'Power BI': ['power bi', 'powerbi', 'power-bi'],
        'Tableau': ['tableau', 'tableau desktop'],
        'Looker': ['looker', 'google looker'],
        'QlikView': ['qlikview', 'qlik view', 'qlik'],
        'SPSS': ['spss', 'ibm spss'],
        'SAS': ['sas', 'sas analytics']
    },
    'Cloud & Big Data': {
        'AWS': ['aws', 'amazon web services', 'ec2', 's3', 'redshift'],
        'Azure': ['azure', 'microsoft azure', 'azure data factory'],
        'GCP': ['gcp', 'google cloud', 'bigquery', 'google cloud platform'],
        'Hadoop': ['hadoop', 'hdfs', 'mapreduce'],
        'Spark': ['apache spark', 'spark', 'pyspark'],
        'Snowflake': ['snowflake'],
        'Databricks': ['databricks']
    },
    'Machine Learning & AI': {
        'Machine Learning': ['machine learning', 'ml', 'scikit-learn', 'sklearn'],
        'Deep Learning': ['deep learning', 'neural networks', 'dl'],
        'TensorFlow': ['tensorflow', 'tf'],
        'PyTorch': ['pytorch', 'torch'],
        'NLP': ['nlp', 'natural language processing', 'text mining'],
        'Computer Vision': ['computer vision', 'cv', 'image processing'],
        'AI': ['artificial intelligence', 'ai']
    },
    'Statistics & Math': {
        'Statistics': ['statistics', 'statistical analysis', 'statistical modeling'],
        'Mathematics': ['mathematics', 'math', 'mathematical modeling'],
        'A/B Testing': ['a/b testing', 'ab testing', 'hypothesis testing'],
        'Regression': ['regression', 'linear regression', 'logistic regression'],
        'Time Series': ['time series', 'forecasting', 'arima']
    },
    'Data Engineering': {
        'ETL': ['etl', 'data pipeline', 'data pipelines'],
        'Docker': ['docker', 'containerization'],
        'Kubernetes': ['kubernetes', 'k8s'],
        'Airflow': ['airflow', 'apache airflow'],
        'Git': ['git', 'github', 'version control'],
        'Linux': ['linux', 'unix', 'bash']
    },
    'Soft Skills': {
        'Communication': ['communication', 'presentation', 'stakeholder management'],
        'Problem Solving': ['problem solving', 'analytical thinking', 'critical thinking'],
        'Project Management': ['project management', 'agile', 'scrum'],
        'Leadership': ['leadership', 'team lead', 'mentoring']
    }
}

def extract_skills_enhanced(text, skill_categories):
    """Enhanced skill extraction with better pattern matching"""
    if pd.isna(text):
        return {}
    
    text = str(text).lower()
    found_skills = defaultdict(list)
    
    for category, skills in skill_categories.items():
        for skill_name, patterns in skills.items():
            for pattern in patterns:
                if re.search(r'\b' + pattern + r'\b', text, re.IGNORECASE):
                    found_skills[category].append(skill_name)
                    break  # Found this skill, move to next
    
    # Remove duplicates while preserving order
    for category in found_skills:
        found_skills[category] = list(dict.fromkeys(found_skills[category]))
    
    return dict(found_skills)

def create_skills_heatmap(df):
    """Create a heatmap showing skills by role"""
    roles = df['role'].value_counts().head(8).index  # Top 8 roles
    
    # Get all unique skills
    all_skills = set()
    for skills_dict in df['found_skills']:
        for category_skills in skills_dict.values():
            all_skills.update(category_skills)
    
    # Create matrix
    skill_role_matrix = []
    skill_names = sorted(list(all_skills))[:15]  # Top 15 skills
    
    for skill in skill_names:
        row = []
        for role in roles:
            role_df = df[df['role'] == role]
            skill_count = sum(1 for skills_dict in role_df['found_skills'] 
                            for category_skills in skills_dict.values() 
                            if skill in category_skills)
            row.append(skill_count)
        skill_role_matrix.append(row)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(skill_role_matrix, 
                xticklabels=[role.replace(' ', '\n') for role in roles],
                yticklabels=skill_names,
                annot=True, 
                fmt='d',
                cmap='YlOrRd',
                cbar_kws={'label': 'Number of Job Mentions'})
    
    plt.title('Skills Demand Heatmap by Job Role', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Job Roles', fontsize=12)
    plt.ylabel('Skills', fontsize=12)
    plt.tight_layout()
    plt.savefig('skills_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_skill_category_chart(df):
    """Create a chart showing demand by skill category"""
    category_counts = Counter()
    
    for skills_dict in df['found_skills']:
        for category, skills in skills_dict.items():
            category_counts[category] += len(skills)
    
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, counts, color=sns.color_palette("husl", len(categories)))
    plt.title('Job Demand by Skill Category', fontsize=16, fontweight='bold')
    plt.xlabel('Skill Categories', fontsize=12)
    plt.ylabel('Total Mentions', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('skill_categories.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_experience_salary_analysis(df):
    """Analyze salary trends if salary data is available"""
    # Extract years of experience from job descriptions
    experience_pattern = r'(\d+)[\s\-\+]*(?:years?|yrs?).{0,20}(?:experience|exp)'
    
    df['experience_years'] = df['description'].str.extract(experience_pattern, flags=re.IGNORECASE)[0]
    df['experience_years'] = pd.to_numeric(df['experience_years'], errors='coerce')
    
    # Group by experience ranges
    def experience_range(years):
        if pd.isna(years):
            return 'Not Specified'
        elif years <= 2:
            return '0-2 years'
        elif years <= 5:
            return '3-5 years'
        elif years <= 10:
            return '6-10 years'
        else:
            return '10+ years'
    
    df['experience_range'] = df['experience_years'].apply(experience_range)
    
    exp_counts = df['experience_range'].value_counts()
    
    plt.figure(figsize=(10, 6))
    plt.pie(exp_counts.values, labels=exp_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=sns.color_palette("pastel"))
    plt.title('Job Distribution by Experience Level', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.savefig('experience_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_top_companies_chart(df):
    """Show top hiring companies"""
    top_companies = df['company'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(top_companies)), top_companies.values, 
                    color=sns.color_palette("viridis", len(top_companies)))
    plt.yticks(range(len(top_companies)), top_companies.index)
    plt.xlabel('Number of Job Postings', fontsize=12)
    plt.title('Top 10 Companies by Job Postings', fontsize=16, fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(top_companies.values):
        plt.text(v + max(top_companies.values)*0.01, i, str(v), 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('top_companies.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_wordcloud(df):
    """Create a word cloud from job descriptions"""
    # Combine all descriptions
    all_text = ' '.join(df['description'].dropna())
    
    # Remove common stop words and add domain-specific ones
    stopwords = set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                     'will', 'be', 'have', 'has', 'is', 'are', 'was', 'were', 'been', 'being',
                     'candidate', 'experience', 'work', 'team', 'role', 'job', 'position', 'looking'])
    
    wordcloud = WordCloud(width=1200, height=600, 
                         background_color='white',
                         stopwords=stopwords,
                         max_words=100,
                         colormap='viridis').generate(all_text)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Job Descriptions', fontsize=20, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('job_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print(f"Loaded {len(df)} job postings")
    print(f"Columns: {list(df.columns)}")
    
    # Enhanced skill extraction
    print("Extracting skills with enhanced detection...")
    df['found_skills'] = df['description'].apply(lambda x: extract_skills_enhanced(x, SKILL_CATEGORIES))
    
    # Calculate some statistics
    jobs_with_skills = sum(1 for skills_dict in df['found_skills'] if any(skills_dict.values()))
    print(f"Jobs with identified skills: {jobs_with_skills}/{len(df)} ({jobs_with_skills/len(df)*100:.1f}%)")
    
    # Create visualizations
    create_skill_category_chart(df)
    create_skills_heatmap(df)
    create_experience_salary_analysis(df)
    create_top_companies_chart(df)
    
    # Only create wordcloud if wordcloud is installed
    try:
        create_wordcloud(df)
    except ImportError:
        print("WordCloud not installed. Skipping word cloud generation.")
        print("Install with: pip install wordcloud")
    
    # Summary statistics
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    all_skills = []
    for skills_dict in df['found_skills']:
        for category_skills in skills_dict.values():
            all_skills.extend(category_skills)
    
    if all_skills:
        skill_counter = Counter(all_skills)
        print(f"Most in-demand skills:")
        for skill, count in skill_counter.most_common(5):
            print(f"  {skill}: {count} mentions ({count/len(df)*100:.1f}% of jobs)")
    
    print(f"\nTop locations:")
    for location, count in df['location'].value_counts().head(5).items():
        print(f"  {location}: {count} jobs")
    
    print(f"\nTop roles:")
    for role, count in df['role'].value_counts().head(5).items():
        print(f"  {role}: {count} jobs")

if __name__ == "__main__":
    main()