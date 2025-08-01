import pandas as pd
from collections import Counter

# Load your scraped data
df = pd.read_csv("adzuna_all_jobs.csv")

# Skill list to search for
SKILLS = [
    "Python", "SQL", "Excel", "Power BI", "Tableau", "R", "Java",
    "Communication", "Problem Solving", "Machine Learning", "Deep Learning",
    "Statistics", "NLP", "ETL", "AWS", "Azure", "Git", "Linux"
]

# Clean descriptions and search for skill mentions
def extract_skills(text, skills):
    text = str(text).lower()
    found = []
    for skill in skills:
        if skill.lower() in text:
            found.append(skill)
    return found

# Apply skill extraction
df["found_skills"] = df["description"].apply(lambda x: extract_skills(x, SKILLS))

# Flatten all skills into a Counter
all_skills = sum(df["found_skills"].tolist(), [])
skill_counts = Counter(all_skills)

# Show top skills
print("\n🔥 Top Skills Across All Roles:\n")
for skill, count in skill_counts.most_common(15):
    print(f"{skill}: {count}")
