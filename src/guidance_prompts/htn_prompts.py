import os
from LLM_api import call_groq_api

def is_granular(task, capabilities_input):
    prompt = f"""Given the capabilities {capabilities_input}, is the task '{task}' granular enough to be directly executed or considered a primitive action by a robot? For instance, if a task includes the word scan, then it's granular because scanning can't be further broken down. Answer with "Yes" or "No"."""
    
    response = call_groq_api(prompt, strip=True)
    return response == "Yes"

def translate(goal_input, original_task, capabilities_input):
    prompt = f"""Given this parent goal {goal_input}, translate the task '{original_task}' into a form that can be executed by a robot using the following capabilities:
    '{capabilities_input}'. Provide the executable form in a single line without any commentary
    or superfluous text.
    
    When translated to use the specified capabilities the result is:"""

    response = call_groq_api(prompt, strip=True)
    return response

def evaluate_candidate(goal_input, task, subtasks, capabilities_input):
    prompt = f"""Given the parent goal {goal_input}, and the parent task {task}, and its subtasks {subtasks}, 
    evaluate how well these subtasks address the requirements 
    of the parent task without any gaps or redundancies, using the following capabilities: 
    {capabilities_input}
    Return a score between 0 and 1, where 1 is the best possible score.
    
    Please follow this regex expression: ^[0]\.\d{{8}}$
    Score:"""

    response = call_groq_api(prompt, strip=True)
    return response

def check_subtasks(task, subtasks, capabilities_input):
    prompt = f"""Given the parent task '{task}', and its subtasks '{', '.join(subtasks)}',
    check if these subtasks effectively and comprehensively address the requirements
    of the parent task without any gaps or redundancies, using the following capabilities:
    '{capabilities_input}'. Return 'True' if they meet the requirements or 'False' otherwise."""

    response = call_groq_api(prompt, strip=True)
    return response.lower() == 'true'

def get_subtasks(task, state, remaining_decompositions, capabilities_input):
    prompt = f"""Given the task '{task}', the current state '{state}',
    and {remaining_decompositions} decompositions remaining before failing,
    please decompose the task into a detailed step-by-step plan
    that can be achieved using the following capabilities:
    '{capabilities_input}'. Provide the subtasks in a comma-separated list,
    each enclosed in square brackets: [subtask1], [subtask2], ..."""

    response = call_groq_api(prompt, strip=True)
    return response