from text_preprocessor import (
    clean_text,
    extract_sections
)


resume = """
JOHN DOE

Email: john@example.com

SKILLS

Python
SQL
Machine Learning
Deep Learning

EDUCATION

Bachelor of Technology in Computer Science

EXPERIENCE

Python Developer with 2 years experience.

PROJECTS

AI Career Recommendation System
"""


print("========== CLEAN TEXT ==========")

cleaned = clean_text(resume)

print(cleaned)


print("\n========== SECTIONS ==========")

sections = extract_sections(resume)

for section, content in sections.items():

    print(f"\n{section.upper()}:")
    print(content)