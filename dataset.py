import torch
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from lxml.html import fromstring


def load_data(file_name: str, folder_path: str="data"):
    """
    Loads data from a file into a pandas DataFrame.

    Args:
        file_name: str - name of the data file (e.g., .tsv, .txt)
        folder_path: str - path to the folder containing the file

    Returns:
        DataFrame:
            Loaded data as a pandas DataFrame
    """
    return pd.read_table(
        os.path.join(folder_path, file_name),
        on_bad_lines="skip"
    )


def html_to_sbert_text(html: str) -> str:
    """
    Converts raw input into clean plain text

    Args:
        html: str - raw text input

    Returns:
        str:
            Cleaned text with normalized whitespace
    """
    if not html:
        return ""
    html = str(html).strip()
    if not html:
        return ""
    try:
        text = fromstring(html).text_content()
    except Exception:
        text = html # fallback: keep raw text
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def construct_a_prompt_st_linkedin(row):
    """
    Constructs a text prompt for a job from a dataset:
    LinkedIn Job Postings (2023 - 2024)

    Args:
        row: Series/dict - (title, description, salary, experience)

    Returns:
        str:
            Formatted prompt
    """
    title = str(row["title"])
    description = row["description"]
    normalized_salary = row["normalized_salary"]
    formatted_experience_level = row["formatted_experience_level"]
    return f"Title: {title}. Description: {description}. Salary per year: {normalized_salary} dollars. Required experience level: {formatted_experience_level}."

def construct_a_prompt_st(row):
    """
    Builds a cleaned text prompt for a job from a dataset:
    CareerBuilder

    Args:
        row: Series/dict - (Title, Description, Requirements)

    Returns:
        str:
            Formatted prompt
    """
    title = str(row["Title"])
    description = html_to_sbert_text(row["Description"])
    requirements = html_to_sbert_text(row["Requirements"])
    if requirements == "Please refer to the Job Description to view the requirements for this job":
        requirements = None 
    requirements_text = f" Requirements: {requirements}." if requirements else ""
    return f"Title: {title}. Description: {description}.{requirements_text}"

def count_words(pandas): 
    counts = np.zeros((len(pandas),3))
    for index, row in tqdm(pandas.iterrows()):
        title = str(row['Title'])
        description = html_to_sbert_text(row['Description'])
        requirements = html_to_sbert_text(row['Requirements'])
        counts[index][0] = len(title.split())
        counts[index][1] = len(description.split())
        counts[index][2] = len(requirements.split())
    return counts



def user_create_prompt(row, work_df):
    """
    Builds a cleaned text prompt for a user from a dataset:
    CareerBuilder

    Args:
        row: Series/dict - user attributes (education, experience, employment, management)
        work_df: DataFrame - work history data linked by UserID

    Returns:
        str:
            Formatted prompt
    """
    currently_employed = (
        "" if row["CurrentlyEmployed"] is True
        else "not " if row["CurrentlyEmployed"] is False
        else None) 

    managed_clause = (
        f"I have people-management experience leading a team of {int(row['ManagedHowMany']) if float(row['ManagedHowMany']).is_integer() else row['ManagedHowMany']} people."
        if row.get("ManagedHowMany") and not pd.isna(row.get("ManagedHowMany")) and row.get("ManagedHowMany") != 0
        else "I have people-management experience." if row["ManagedOthers"]
        else ".")

    major_clause = (
        f", where I got a major in {row['Major']}"
        if row.get("Major") and not pd.isna(row.get("Major"))
        else "")

    graduation_clause = (
        f" in {pd.to_datetime(row['GraduationDate']).year}"
        if row.get("GraduationDate") and not pd.isna(row.get("GraduationDate"))
        else "")

    education_clause = (
        f"I have a {row['DegreeType']} education{major_clause}{graduation_clause}. "
        if (
            (row.get("DegreeType") and not pd.isna(row.get("DegreeType")) and str(row.get("DegreeType")).strip().lower() != "none")
            or (row.get("Major") and not pd.isna(row.get("Major")))
        )
        else "")

    employment_clause = (
        f"I am {currently_employed}currently employed"
        if currently_employed is not None
        else "")

    work_history_clause = work_history_sentence(work_df, row["UserID"])

    years_experience = (  int(row["TotalYearsExperience"]) if row.get("TotalYearsExperience") is not None and not pd.isna(row.get("TotalYearsExperience")) else "")

    prompt = (
        f"{education_clause}"
        f"I have {years_experience} years of experience and have held {row['WorkHistoryCount']} roles."
        f"{work_history_clause} "
        f"{employment_clause}{managed_clause}")

    return prompt


def user_prompt_basic(row):
    """
    Builds a cleaned text prompt for a user from a dataset:
    CareerBuilder

    Simplified version of a user_create_prompt used for constructing synthetic profiles

    Args:
        row: Series/dict - (education, experience, employment, management)

    Returns:
        str:
            Formatted prompt
    """
    edu = ""
    if row.get("DegreeType") is not None and pd.notna(row.get("DegreeType")) and str(row.get("DegreeType")).strip().lower() != "none":
        edu = f"I have a {row['DegreeType']} education."

    major = ""
    if row.get("Major") is not None and pd.notna(row.get("Major")):
        major = f" My major is {row['Major']}."

    grad = ""
    if row.get("GraduationDate") is not None and pd.notna(row.get("GraduationDate")):
        grad = f" Graduation year: {pd.to_datetime(row['GraduationDate']).year}."

    yrs = ""
    if row.get("TotalYearsExperience") is not None and pd.notna(row.get("TotalYearsExperience")):
        yrs = f" I have {int(float(row['TotalYearsExperience']))} years of experience."

    roles = ""
    if row.get("WorkHistoryCount") is not None and pd.notna(row.get("WorkHistoryCount")):
        roles = f" I have held {int(float(row['WorkHistoryCount']))} roles."

    employed = ""
    if row.get("CurrentlyEmployed") is True:
        employed = " I am currently employed."
    elif row.get("CurrentlyEmployed") is False:
        employed = " I am not currently employed."

    managed = ""
    if row.get("ManagedOthers") is True:
        mhm = row.get("ManagedHowMany")
        if mhm is not None and pd.notna(mhm) and float(mhm) != 0.0:
            managed = f" I have people-management experience leading a team of {int(mhm) if float(mhm).is_integer() else mhm} people."
        else:
            managed = " I have people-management experience."

    prompt = f"{edu}{major}{grad}{yrs}{roles}{employed}{managed}".strip()
    return prompt

def work_history_sentence(
        work_df: pd.DataFrame, 
        user_id: str | int,
        max_titles: int=10
    ):
    """
    Constructs a sentence summarizing a users work history

    Args:
        work_df: DataFrame - contains work history (UserID and JobTitle)
        user_id: str/int - ID user
        max_titles: int - max number of job titles to include

    Returns:
        str:
            list of previous job titles, if none, empty string 
    """
    df = work_df[work_df["UserID"] == user_id]
    df = df.sort_values("Sequence")

    seen = set() 
    titles = [] 
    for t in df["JobTitle"].dropna().astype(str):
        if t.strip() and t not in seen:
            seen.add(t) 
            titles.append(t)  

    titles = titles[:max_titles]
    if not titles:
        return ""

    return " Work history includes: " + "; ".join(titles) + "."



def get_jobs_with_interactions(
        semantics: np.ndarray,
        jobs: pd.DataFrame,
        interactions: pd.DataFrame,
        users: pd.DataFrame,
        users_semantics: np.ndarray
    ):
    """
    Filters jobs and users to only those with interactions.

    Args:
        semantics: np.ndarray - embeddings (jobs)
        jobs: pd.DataFrame - job metadata
        interactions: pd.DataFrame - user-job interactions
        users: pd.DataFrame - user metadata
        users_semantics: np.ndarray - embeddings (all users)

    Returns:
        tuple containing:
            jobs_df: pd.DataFrame - embeddings (valid jobs) sorted
            interactions_clean: pd.DataFrame - interactions with valid users and jobs
            missing_jobids: list - JobIDs missing from jobs
            users_clean: pd.DataFrame - users in cleaned interactions
            missing_userids: list - UserIDs missing from users
            users_df: pd.DataFrame - embeddings (valid users) sorted
    """
    jobs_to_use = pd.unique(interactions["JobID"])
    jobid_to_idx = pd.Series(np.arange(len(jobs)), index=jobs["JobID"]).to_dict()
    missing_jobids = [j for j in jobs_to_use if j not in jobid_to_idx]
    interactions_clean = interactions[interactions["JobID"].isin(jobid_to_idx)] 
    valid_jobids = pd.unique(interactions_clean["JobID"]) 
    indecies_to_use = [jobid_to_idx[j] for j in valid_jobids]
    if len(valid_jobids) > 0:
        jid = valid_jobids[0] 
        idx = jobid_to_idx[jid]
        if idx >= len(semantics):
            raise IndexError(f"Sanity check failed: idx {idx} out of bounds for semantics (size {len(semantics)}).")
        if jobs.iloc[idx]["JobID"] != jid: 
            raise ValueError(f"Sanity check failed: jobs.iloc[{idx}]['JobID']={jobs.iloc[idx]['JobID']} != {jid}.") 
    new_semantics = semantics[indecies_to_use]
    jobs_df = pd.DataFrame(new_semantics, index=valid_jobids)

    users_to_use = pd.unique(interactions_clean["UserID"])
    userid_to_idx = pd.Series(np.arange(len(users)), index=users["UserID"]).to_dict()
    missing_userids = [u for u in users_to_use if u not in userid_to_idx]
    interactions_clean = interactions_clean[interactions_clean["UserID"].isin(userid_to_idx)]
    valid_userids = pd.unique(interactions_clean["UserID"])
    user_indecies_to_use = [userid_to_idx[u] for u in valid_userids]

    if len(valid_userids) > 0:
        uid = valid_userids[0]
        uidx = userid_to_idx[uid]
        if uidx >= len(users_semantics):
            raise IndexError(f"User sanity check failed: idx {uidx} out of bounds for users_semantics (size {len(users_semantics)}).")
        if users.iloc[uidx]["UserID"] != uid:
            raise ValueError(f"User sanity check failed: users.iloc[{uidx}]['UserID']={users.iloc[uidx]['UserID']} != {uid}.")

    new_users_semantics = users_semantics[user_indecies_to_use]
    users_df = pd.DataFrame(new_users_semantics, index=valid_userids)
    users_clean = users[users["UserID"].isin(valid_userids)]

    return jobs_df, interactions_clean, missing_jobids, users_clean, missing_userids, users_df

class UBERTDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for user-job interaction

    Args:
        jobs: pd.DataFrame - embeddings (valid jobs) indexed by id
        interactions: pd.DataFrame - interactions with valid users and jobs
        users: pd.DataFrame - embeddings (valid users) indexed by id
    """
    def __init__(self, jobs, interactions, users):
        self.interactions = interactions
        self.jobs = jobs
        self.users = users

        
    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx: int):
        user_id = self.interactions.iloc[idx]['UserID']
        job_id = self.interactions.iloc[idx]['JobID']
        user_vec = torch.as_tensor(self.users.loc[user_id].values, dtype=torch.float32)
        job_vec = torch.as_tensor(self.jobs.loc[job_id].values, dtype=torch.float32)
        return user_vec, job_vec
