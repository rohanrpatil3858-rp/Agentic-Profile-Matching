"""
generate_dataset.py
====================

Generate a synthetic but realistic test dataset for the
Agentic Profile Matching project:

* 35 fictional resumes as PDF files in ``data/resumes/``
* 6 job descriptions as ``.txt`` files in ``data/job_descriptions/``

All candidates and companies are entirely fictional. The generator is
**reproducible** (fixed random seed) and creates output directories as needed.

Resumes are intentionally diverse across roles, experience levels (~1-12 years)
and skill combinations so the hybrid retrieval / ranking system can be tested
with strong, partial and poor matches.

Usage
-----
Run from the project root so the relative ``data/`` paths resolve correctly::

    python scripts/generate_dataset.py
"""

from __future__ import annotations

import os
import random
from typing import Dict, List

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem,
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RANDOM_SEED = 42
NUM_RESUMES = 35

RESUME_DIR = os.path.join("data", "resumes")
JD_DIR = os.path.join("data", "job_descriptions")


# --------------------------------------------------------------------------- #
# Fictional name pools (generic, not tied to real individuals)
# --------------------------------------------------------------------------- #

FIRST_NAMES = [
    "Ava", "Liam", "Noah", "Mia", "Ethan", "Zoe", "Lucas", "Aria", "Leo",
    "Isla", "Kai", "Nora", "Ravi", "Priya", "Wei", "Sofia", "Diego", "Amara",
    "Yuki", "Omar", "Elena", "Marcus", "Ingrid", "Tariq", "Hana", "Sven",
    "Lena", "Mateo", "Freya", "Arjun", "Nadia", "Felix", "Sana", "Bruno",
    "Camila",
]

LAST_NAMES = [
    "Bennett", "Carver", "Dalton", "Ellison", "Frost", "Grover", "Hale",
    "Ibarra", "Juno", "Keller", "Lindgren", "Marsh", "Novak", "Okafor",
    "Pereira", "Quinn", "Rhodes", "Salazar", "Takeda", "Underwood", "Voss",
    "Whitfield", "Xu", "Yates", "Zimmer", "Ashford", "Blackwood", "Castellan",
    "Delacroix", "Eastwood", "Fairbanks", "Ghosh", "Halvorsen", "Ionescu",
    "Jansson",
]

FICTIONAL_COMPANIES = [
    "Nimbus Labs", "Quantic Systems", "BluePeak Analytics", "Corevance",
    "Orbit Data", "Helios AI", "Meridian Soft", "Northwind Tech",
    "Vertex Cloud", "Palewave", "Ironleaf", "Sundial Robotics",
    "Cobalt Metrics", "Everline", "Brightforge",
]

UNIVERSITIES = [
    "Northgate University", "Westbrook Institute of Technology",
    "Riverdale State University", "Ashcombe College",
    "Grandview Technical University", "Fairhaven University",
    "Summit Polytechnic", "Lakemoor University",
]


# --------------------------------------------------------------------------- #
# Role definitions: skill pools + section text templates
# --------------------------------------------------------------------------- #

# Each role has a set of "core" skills (strong signal) and "adjacent" skills.
ROLES: Dict[str, Dict[str, List[str]]] = {
    "Machine Learning Engineer": {
        "core": ["Python", "Machine Learning", "Deep Learning", "PyTorch",
                 "TensorFlow", "NLP", "scikit-learn", "NumPy", "Pandas"],
        "adjacent": ["MLOps", "Docker", "AWS", "Kubernetes", "SQL",
                     "Hugging Face", "LLM", "RAG", "Airflow"],
        "degree": "M.Sc. in Computer Science",
    },
    "Data Scientist": {
        "core": ["Python", "Statistics", "Machine Learning", "Pandas",
                 "NumPy", "scikit-learn", "Data Analysis", "SQL"],
        "adjacent": ["R", "Tableau", "Deep Learning", "Spark", "Matplotlib",
                     "Power BI", "NLP", "A/B testing"],
        "degree": "M.Sc. in Data Science",
    },
    "Data Analyst": {
        "core": ["SQL", "Excel", "Data Analysis", "Tableau", "Power BI",
                 "Python", "Statistics"],
        "adjacent": ["Pandas", "R", "ETL", "Data Visualization",
                     "Google Analytics", "Looker"],
        "degree": "B.Sc. in Statistics",
    },
    "Python Developer": {
        "core": ["Python", "Django", "Flask", "FastAPI", "REST", "SQL",
                 "PostgreSQL"],
        "adjacent": ["Docker", "Redis", "Celery", "AWS", "GraphQL", "pytest",
                     "Microservices"],
        "degree": "B.Tech in Computer Engineering",
    },
    "Backend Engineer": {
        "core": ["Java", "Spring Boot", "Microservices", "REST", "SQL",
                 "PostgreSQL", "Kafka"],
        "adjacent": ["Docker", "Kubernetes", "Redis", "gRPC", "AWS", "Go",
                     "MongoDB"],
        "degree": "B.Sc. in Computer Science",
    },
    "Frontend Engineer": {
        "core": ["JavaScript", "TypeScript", "React", "HTML", "CSS",
                 "Redux", "Next.js"],
        "adjacent": ["Vue", "GraphQL", "Jest", "Webpack", "Tailwind",
                     "Accessibility", "Figma"],
        "degree": "B.Sc. in Software Engineering",
    },
    "Full Stack Developer": {
        "core": ["JavaScript", "TypeScript", "React", "Node.js", "Express",
                 "MongoDB", "REST", "SQL"],
        "adjacent": ["Python", "Docker", "AWS", "GraphQL", "PostgreSQL",
                     "Next.js", "CI/CD"],
        "degree": "B.Tech in Information Technology",
    },
    "DevOps Engineer": {
        "core": ["Docker", "Kubernetes", "CI/CD", "Terraform", "Jenkins",
                 "Linux", "AWS", "Ansible"],
        "adjacent": ["Python", "GCP", "Azure", "Prometheus", "Bash",
                     "GitOps", "Monitoring"],
        "degree": "B.Sc. in Computer Science",
    },
    "Cloud Engineer": {
        "core": ["AWS", "GCP", "Azure", "Terraform", "Kubernetes", "Docker",
                 "Linux", "Networking"],
        "adjacent": ["Python", "CI/CD", "Serverless", "CloudFormation",
                     "Security", "Ansible"],
        "degree": "B.Sc. in Information Systems",
    },
    "Product Manager": {
        "core": ["Product Strategy", "Roadmapping", "Agile", "Scrum",
                 "Stakeholder Management", "Analytics", "User Research"],
        "adjacent": ["SQL", "A/B testing", "Jira", "Figma", "Data Analysis",
                     "Go-to-Market"],
        "degree": "MBA",
    },
}

CERTIFICATIONS_BY_ROLE: Dict[str, List[str]] = {
    "Machine Learning Engineer": ["TensorFlow Developer Certificate",
                                  "AWS Certified Machine Learning – Specialty"],
    "Data Scientist": ["Google Data Analytics Certificate",
                       "Databricks Certified ML Associate"],
    "Data Analyst": ["Microsoft Power BI Data Analyst Associate",
                     "Tableau Desktop Specialist"],
    "Python Developer": ["PCEP – Certified Entry-Level Python Programmer"],
    "Backend Engineer": ["Oracle Certified Professional: Java SE",
                         "Confluent Certified Developer for Apache Kafka"],
    "Frontend Engineer": ["Meta Front-End Developer Certificate"],
    "Full Stack Developer": ["AWS Certified Developer – Associate"],
    "DevOps Engineer": ["Certified Kubernetes Administrator (CKA)",
                        "HashiCorp Certified: Terraform Associate"],
    "Cloud Engineer": ["AWS Certified Solutions Architect – Associate",
                       "Google Associate Cloud Engineer"],
    "Product Manager": ["Certified Scrum Product Owner (CSPO)"],
}

# Project blurbs per role (templated with fictional companies/skills).
PROJECT_TEMPLATES: Dict[str, List[str]] = {
    "Machine Learning Engineer": [
        "Built a document retrieval and RAG pipeline using {skill_a} and "
        "{skill_b}, improving answer relevance by 27%.",
        "Trained and deployed an NLP classification model with {skill_a}, "
        "serving 2M+ requests/day.",
    ],
    "Data Scientist": [
        "Developed a churn prediction model in {skill_a} that reduced customer "
        "attrition by 15%.",
        "Designed an A/B testing framework and dashboards using {skill_b}.",
    ],
    "Data Analyst": [
        "Automated weekly reporting with {skill_a} and {skill_b}, saving "
        "10 hours/week.",
        "Built executive dashboards in {skill_b} tracking core KPIs.",
    ],
    "Python Developer": [
        "Designed a REST API with {skill_a} handling 500 req/s with 99.9% "
        "uptime.",
        "Migrated a monolith to microservices using {skill_a} and {skill_b}.",
    ],
    "Backend Engineer": [
        "Implemented an event-driven order system with {skill_a} and {skill_b}.",
        "Optimized database queries reducing p95 latency by 40%.",
    ],
    "Frontend Engineer": [
        "Rebuilt the customer portal in {skill_a} improving Lighthouse score "
        "to 98.",
        "Created a reusable component library with {skill_a} and {skill_b}.",
    ],
    "Full Stack Developer": [
        "Shipped a full-stack marketplace using {skill_a} and {skill_b}.",
        "Built real-time chat with WebSockets on a {skill_a}/{skill_b} stack.",
    ],
    "DevOps Engineer": [
        "Automated CI/CD with {skill_a} and {skill_b}, cutting deploy time "
        "by 60%.",
        "Provisioned infrastructure as code using {skill_a}.",
    ],
    "Cloud Engineer": [
        "Migrated on-prem workloads to {skill_a}, reducing costs by 35%.",
        "Designed multi-region architecture with {skill_a} and {skill_b}.",
    ],
    "Product Manager": [
        "Led the launch of a new analytics product, driving 20% MoM growth.",
        "Ran discovery and user research to reprioritize the roadmap.",
    ],
}

# Role frequency: emphasise the roles that the 6 JDs target so we get a
# realistic mix of strong / partial / poor matches.
ROLE_DISTRIBUTION = [
    "Machine Learning Engineer", "Machine Learning Engineer",
    "Machine Learning Engineer", "Machine Learning Engineer",
    "Data Scientist", "Data Scientist", "Data Scientist",
    "Data Analyst", "Data Analyst", "Data Analyst", "Data Analyst",
    "Python Developer", "Python Developer", "Python Developer",
    "Python Developer",
    "Backend Engineer", "Backend Engineer", "Backend Engineer",
    "Frontend Engineer", "Frontend Engineer", "Frontend Engineer",
    "Frontend Engineer",
    "Full Stack Developer", "Full Stack Developer", "Full Stack Developer",
    "Full Stack Developer",
    "DevOps Engineer", "DevOps Engineer", "DevOps Engineer",
    "Cloud Engineer", "Cloud Engineer", "Cloud Engineer",
    "Product Manager", "Product Manager", "Product Manager",
]


# --------------------------------------------------------------------------- #
# Resume content generation
# --------------------------------------------------------------------------- #

def _pick_skills(role: str, experience: int) -> List[str]:
    """Choose a diverse skill set for a candidate.

    More experienced candidates get more skills. We always include a subset of
    core skills (so strong matches exist) and a random subset of adjacent
    skills (so partial matches and variety exist).
    """
    role_data = ROLES[role]
    core = role_data["core"]
    adjacent = role_data["adjacent"]

    # Number of core skills scales with experience (min 3).
    n_core = min(len(core), max(3, 3 + experience // 3))
    n_adjacent = min(len(adjacent), random.randint(1, 4))

    chosen = random.sample(core, n_core) + random.sample(adjacent, n_adjacent)
    # Deduplicate while preserving order.
    seen, result = set(), []
    for skill in chosen:
        if skill not in seen:
            seen.add(skill)
            result.append(skill)
    return result


def _build_experience_entries(
    role: str, experience: int, skills: List[str]
) -> List[str]:
    """Create 1-3 work-experience bullet lines summing near `experience` years."""
    entries: List[str] = []
    remaining = experience
    num_jobs = min(3, max(1, experience // 3 + 1))
    end_year = 2026

    for i in range(num_jobs):
        # Allocate a plausible tenure per role.
        if i == num_jobs - 1:
            tenure = max(1, remaining)
        else:
            tenure = max(1, min(remaining - (num_jobs - i - 1), random.randint(1, 4)))
        remaining -= tenure
        start_year = end_year - tenure
        company = random.choice(FICTIONAL_COMPANIES)
        title = role if i == 0 else f"{'Senior ' if experience > 6 else ''}{role}"
        skill_a = skills[0] if skills else "Python"
        skill_b = skills[1] if len(skills) > 1 else skill_a
        entries.append(
            f"<b>{title}</b>, {company} ({start_year}\u2013{end_year})<br/>"
            f"Delivered projects using {skill_a} and {skill_b}; "
            f"collaborated in an Agile team to ship features to production."
        )
        end_year = start_year
        if remaining <= 0:
            break

    return entries


def _build_summary(name: str, role: str, experience: int,
                   skills: List[str]) -> str:
    """Compose a professional summary paragraph."""
    level = (
        "Senior" if experience >= 8 else
        "Mid-level" if experience >= 4 else
        "Junior"
    )
    top_skills = ", ".join(skills[:4])
    return (
        f"{level} {role} with {experience} years of experience specializing in "
        f"{top_skills}. Proven track record of delivering production systems and "
        f"collaborating across cross-functional teams."
    )


def _build_projects(role: str, skills: List[str]) -> List[str]:
    """Render project bullet strings for the role."""
    templates = PROJECT_TEMPLATES[role]
    skill_a = skills[0] if skills else "Python"
    skill_b = skills[1] if len(skills) > 1 else skill_a
    return [t.format(skill_a=skill_a, skill_b=skill_b) for t in templates]


def _candidate_record(index: int) -> Dict:
    """Assemble a full candidate record (deterministic given the seed)."""
    role = ROLE_DISTRIBUTION[index % len(ROLE_DISTRIBUTION)]
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    experience = random.randint(1, 12)
    skills = _pick_skills(role, experience)
    role_data = ROLES[role]

    email = f"{first.lower()}.{last.lower()}@example.com"
    phone = f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
    grad_year = 2026 - experience - random.randint(0, 2)
    university = random.choice(UNIVERSITIES)

    certs = CERTIFICATIONS_BY_ROLE.get(role, [])
    # Not everyone has certifications.
    if certs and random.random() < 0.4:
        certs = random.sample(certs, min(len(certs), random.randint(1, len(certs))))
    else:
        certs = certs[:1] if certs and random.random() < 0.6 else []

    return {
        "name": name,
        "first": first,
        "last": last,
        "role": role,
        "experience": experience,
        "skills": skills,
        "email": email,
        "phone": phone,
        "summary": _build_summary(name, role, experience, skills),
        "experience_entries": _build_experience_entries(role, experience, skills),
        "education": f"{role_data['degree']}, {university} ({grad_year})",
        "projects": _build_projects(role, skills),
        "certifications": certs,
    }


# --------------------------------------------------------------------------- #
# PDF rendering
# --------------------------------------------------------------------------- #

def _styles():
    """Build the paragraph styles used across resumes."""
    base = getSampleStyleSheet()
    styles = {
        "name": ParagraphStyle(
            "Name", parent=base["Title"], fontSize=20, spaceAfter=2,
            alignment=TA_LEFT,
        ),
        "contact": ParagraphStyle(
            "Contact", parent=base["Normal"], fontSize=9, textColor="#555555",
            spaceAfter=10,
        ),
        "section": ParagraphStyle(
            "SectionHeader", parent=base["Heading2"], fontSize=12,
            textColor="#1a3c5a", spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"], fontSize=10, leading=14,
        ),
    }
    return styles


def _bullet_list(items: List[str], style) -> ListFlowable:
    """Build a bulleted list flowable from strings."""
    return ListFlowable(
        [ListItem(Paragraph(item, style), leftIndent=6) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=12,
    )


def render_resume_pdf(record: Dict, out_path: str) -> None:
    """Render a single candidate record to a PDF file at ``out_path``.

    Section headers use plain names (Summary, Skills, Experience, Education,
    Projects, Certifications) so the section-aware chunker can detect them.
    """
    styles = _styles()
    doc = SimpleDocTemplate(
        out_path, pagesize=LETTER,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title=record["name"],
    )

    flow = []
    flow.append(Paragraph(record["name"], styles["name"]))
    flow.append(Paragraph(
        f"{record['role']} &nbsp;|&nbsp; {record['email']} "
        f"&nbsp;|&nbsp; {record['phone']}",
        styles["contact"],
    ))

    # Summary
    flow.append(Paragraph("Summary", styles["section"]))
    flow.append(Paragraph(record["summary"], styles["body"]))

    # Skills
    flow.append(Paragraph("Skills", styles["section"]))
    flow.append(Paragraph(", ".join(record["skills"]), styles["body"]))

    # Experience
    flow.append(Paragraph("Experience", styles["section"]))
    flow.append(_bullet_list(record["experience_entries"], styles["body"]))

    # Education
    flow.append(Paragraph("Education", styles["section"]))
    flow.append(Paragraph(record["education"], styles["body"]))

    # Projects
    flow.append(Paragraph("Projects", styles["section"]))
    flow.append(_bullet_list(record["projects"], styles["body"]))

    # Certifications (optional)
    if record["certifications"]:
        flow.append(Paragraph("Certifications", styles["section"]))
        flow.append(_bullet_list(record["certifications"], styles["body"]))

    flow.append(Spacer(1, 6))
    doc.build(flow)


# --------------------------------------------------------------------------- #
# Job description generation
# --------------------------------------------------------------------------- #

JOB_DESCRIPTIONS: List[Dict] = [
    {
        "filename": "machine_learning_engineer.txt",
        "title": "Machine Learning Engineer",
        "min_experience": 4,
        "responsibilities": [
            "Design, train and deploy ML models to production.",
            "Build NLP and retrieval-augmented generation (RAG) pipelines.",
            "Collaborate with data engineering on data quality and pipelines.",
        ],
        "required": ["Python", "Machine Learning", "Deep Learning", "PyTorch",
                     "NLP"],
        "preferred": ["TensorFlow", "Hugging Face", "LLM", "Docker", "AWS",
                      "MLOps"],
    },
    {
        "filename": "senior_python_backend_developer.txt",
        "title": "Senior Python Backend Developer",
        "min_experience": 5,
        "responsibilities": [
            "Build and maintain scalable REST APIs and microservices.",
            "Design database schemas and optimize query performance.",
            "Mentor junior engineers and drive code quality.",
        ],
        "required": ["Python", "Django", "FastAPI", "REST", "PostgreSQL",
                     "SQL"],
        "preferred": ["Docker", "Redis", "Celery", "AWS", "Microservices"],
    },
    {
        "filename": "data_analyst.txt",
        "title": "Data Analyst",
        "min_experience": 2,
        "responsibilities": [
            "Build dashboards and reports for business stakeholders.",
            "Perform ad-hoc analysis to answer product questions.",
            "Define and track key performance metrics.",
        ],
        "required": ["SQL", "Excel", "Tableau", "Data Analysis", "Statistics"],
        "preferred": ["Python", "Power BI", "Pandas", "R", "ETL"],
    },
    {
        "filename": "frontend_developer.txt",
        "title": "Frontend Developer",
        "min_experience": 3,
        "responsibilities": [
            "Develop responsive, accessible web interfaces.",
            "Build reusable UI components and design systems.",
            "Optimize front-end performance and user experience.",
        ],
        "required": ["JavaScript", "TypeScript", "React", "HTML", "CSS"],
        "preferred": ["Next.js", "Redux", "Jest", "Tailwind", "GraphQL"],
    },
    {
        "filename": "devops_cloud_engineer.txt",
        "title": "DevOps / Cloud Engineer",
        "min_experience": 4,
        "responsibilities": [
            "Automate CI/CD pipelines and infrastructure provisioning.",
            "Manage container orchestration and cloud infrastructure.",
            "Implement monitoring, logging and incident response.",
        ],
        "required": ["Docker", "Kubernetes", "CI/CD", "Terraform", "AWS",
                     "Linux"],
        "preferred": ["GCP", "Azure", "Ansible", "Jenkins", "Prometheus",
                      "Python"],
    },
    {
        "filename": "full_stack_developer.txt",
        "title": "Full Stack Developer",
        "min_experience": 3,
        "responsibilities": [
            "Build end-to-end features across front-end and back-end.",
            "Design REST/GraphQL APIs and integrate with databases.",
            "Own features from design through deployment.",
        ],
        "required": ["JavaScript", "TypeScript", "React", "Node.js", "Express",
                     "MongoDB"],
        "preferred": ["Python", "PostgreSQL", "Docker", "AWS", "GraphQL",
                      "CI/CD"],
    },
]


def render_job_description(jd: Dict) -> str:
    """Render a job description dict into formatted plain text."""
    lines: List[str] = []
    lines.append(jd["title"])
    lines.append("")
    lines.append(f"Minimum experience: {jd['min_experience']}+ years")
    lines.append("")
    lines.append("Responsibilities:")
    for item in jd["responsibilities"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Required skills:")
    for item in jd["required"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Preferred skills:")
    for item in jd["preferred"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def generate_resumes(count: int = NUM_RESUMES) -> List[str]:
    """Generate ``count`` resume PDFs and return their paths."""
    os.makedirs(RESUME_DIR, exist_ok=True)
    paths: List[str] = []
    for i in range(count):
        record = _candidate_record(i)
        # Deterministic, unique, filesystem-safe filename.
        slug = f"{record['first']}_{record['last']}".lower()
        filename = f"{i + 1:02d}_{slug}.pdf"
        out_path = os.path.join(RESUME_DIR, filename)
        render_resume_pdf(record, out_path)
        paths.append(out_path)
    return paths


def generate_job_descriptions() -> List[str]:
    """Write all job description ``.txt`` files and return their paths."""
    os.makedirs(JD_DIR, exist_ok=True)
    paths: List[str] = []
    for jd in JOB_DESCRIPTIONS:
        out_path = os.path.join(JD_DIR, jd["filename"])
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(render_job_description(jd))
        paths.append(out_path)
    return paths


def main() -> None:
    """Generate the full dataset reproducibly and print a summary."""
    random.seed(RANDOM_SEED)

    resume_paths = generate_resumes(NUM_RESUMES)
    jd_paths = generate_job_descriptions()

    print("=" * 60)
    print("Dataset generation complete")
    print("=" * 60)
    print(f"Resumes created         : {len(resume_paths)}")
    print(f"Job descriptions created: {len(jd_paths)}")
    print()
    print(f"Resumes saved in        : {os.path.abspath(RESUME_DIR)}")
    for p in resume_paths:
        print(f"  - {p}")
    print()
    print(f"Job descriptions saved in: {os.path.abspath(JD_DIR)}")
    for p in jd_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
