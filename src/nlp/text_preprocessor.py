import re


def clean_text(text: str) -> str:

    # Convert to lowercase
    text = text.lower()

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-z0-9\s+#.-]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def extract_sections(text: str):

    sections = {
        "skills": "",
        "education": "",
        "experience": "",
        "projects": "",
    }

    current_section = None

    section_headers = {
        "skills": "skills",
        "technical skills": "skills",
        "education": "education",
        "experience": "experience",
        "work experience": "experience",
        "projects": "projects",
        "academic projects": "projects",
    }

    lines = text.split("\n")

    for line in lines:

        line = line.strip()

        if not line:
            continue

        lower_line = line.lower()

        # Check whether the entire line is a section heading
        if lower_line in section_headers:

            current_section = section_headers[lower_line]

            continue

        # Add content to current section
        if current_section:

            sections[current_section] += (
                line + " "
            )

    return {
        key: clean_text(value)
        for key, value in sections.items()
    }