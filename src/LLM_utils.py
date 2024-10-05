import datetime
import os
from LLM_api import call_groq_api, log_response
from text_utils import trace_function_calls
from nltk.stem import WordNetLemmatizer

@trace_function_calls
def groq_is_goal(state, goal_task):
    prompt = (f"Given the current state '{state}' and the goal '{goal_task}', "
              f"determine if the current state satisfies the goal. "
              f"Please provide the answer as 'True' or 'False':")

    response = call_groq_api(prompt, strip=True)

    log_response("groq_is_goal", response)
    return response.lower() == "true"

@trace_function_calls
def get_initial_task(goal):
    prompt = f"Given the goal '{goal}', suggest a high level task that will complete it:"

    response = call_groq_api(prompt, strip=True)
    log_response("get_initial_task", response)
    return response

@trace_function_calls
def is_task_primitive(task_name):
    lemmatizer = WordNetLemmatizer()
    task_words = task_name.lower().split()
    
    primitive_actions_keywords = [
        'grab', 'reach', 'twist', 'move', 'push', 'pull', 'lift', 'hold',
        'release', 'turn', 'rotate', 'locate', 'identify', 'find', 'pick',
        'place', 'put', 'insert', 'remove', 'open', 'close', 'clean',
        'wipe', 'sweep', 'mop', 'vacuum', 'wash', 'rinse', 'cook', 'heat',
        'boil', 'fry', 'bake', 'microwave', 'cut', 'slice', 'dice', 'chop', 'examine',
        'grate', 'peel', 'mix', 'blend', 'stir', 'pour', 'serve', 'stop', 'scan', 'activate'
    ]

    for word in task_words:
        lemma = lemmatizer.lemmatize(word)
        if lemma in primitive_actions_keywords:
            return True
    
    return False

@trace_function_calls
def compress_capabilities(text):
    prompt = f"Compress the capabilities description '{text}' into a more concise form:"
    response = call_groq_api(prompt, strip=True)
    return response

@trace_function_calls
def can_execute(task, capabilities, state):
    prompt = (f"Given the task '{task}', the capabilities '{capabilities}', "
              f"and the state '{state}', determine if the task can be executed. "
              f"Please provide the answer as 'True' or 'False':")

    response = call_groq_api(prompt, strip=True)

    log_response("can_execute", response)
    return response.lower() == "true"

def log_state_change(prev_state, new_state, task):
    log_dir = "../state_changes"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = f"{log_dir}/state_changes.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file_path, "a") as log_file:
        log_file.write(f"{timestamp}: Executing task '{task}'\n")
        log_file.write(f"{timestamp}: Previous state: '{prev_state}'\n")
        log_file.write(f"{timestamp}: New state: '{new_state}'\n\n")