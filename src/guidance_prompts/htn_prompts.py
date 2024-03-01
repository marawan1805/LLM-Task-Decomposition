import guidance
import os

guidance_gpt4_api = guidance.llms.OpenAI("gpt-4", api_key=os.environ.get('OPENAI_KEY'))
guidance.llm = guidance_gpt4_api

def is_granular(task, capabilities_input):
    # Check if the given task/subtask is granular enough to be either executed directly or checked if it's primitive.
    granular_check = guidance('''
    {{#system}}You are a helpful agent{{/system}}
    {{#user}}Given the capabilities {{capabilities_input}}, is the task '{{task}}' granular enough to be directly executed or considered a primitive action by a robot? For instance, if a task includes the word scan, then it's granular because scanning can't be further broken down. Anser with "Yes" or "No".{{/user}}
    {{#assistant}}{{gen "granularity_response"}}{{/assistant}}
    ''', llm=guidance_gpt4_api)
    
    result = granular_check(capabilities_input=capabilities_input, task=task)
    return result['granularity_response'] == "Yes"
    

def translate(goal_input, original_task, capabilities_input):
    # translates a task into a form that can be completed with the specified capabilities
    task_translation = guidance('''
    {{#system}}You are a helpful agent{{/system}}
    
    {{#user}}Given this parent goal {{goal_input}}, translate the task '{{task}}' into a form that can be executed by a robot using the following capabilities:
    '{{capabilities_input}}'. Provide the executable form in a single line without any commentary
    or superfluous text.
    
    When translated to use the specified capabilities the result is:{{/user}}
    {{#assistant}}{{gen "translated_task"}}{{/assistant}}
    ''', llm=guidance_gpt4_api)

    result = task_translation(goal_input=goal_input, task=original_task, capabilities_input=capabilities_input)
    return result['translated_task']


def evaluate_candidate(goal_input, task, subtasks, capabilities_input):
    evaluation = guidance('''
    {{#system}}You are a helpful agent{{/system}}

    {{#user}}
    Given the parent goal {{goal_input}}, and the parent task {{task}}, and its subtasks {{subtasks}}, 
    evaluate how well these subtasks address the requirements 
    of the parent task without any gaps or redundancies, using the following capabilities: 
    {{capabilities_input}}
    Return a score between 0 and 1, where 1 is the best possible score.
    
    Please follow this regex expression: ^[0]\.\d{8}$
    Score:
    {{/user}}
    {{#assistant~}}
    {{gen 'score' temperature=0.5 max_tokens=10}}
    {{~/assistant}}''',
                          llm=guidance_gpt4_api)

    result = evaluation(goal_input=goal_input, task=task, subtasks=subtasks, capabilities_input=capabilities_input)
    return result['score'] if 'score' in result else result['Score']

def check_subtasks(task, subtasks, capabilities_input):
    check_subtasks_program = guidance('''
    {{#system~}}
    You are a helpful assistant.
    {{~/system}}
    {{#user~}}
    Given the parent task '{{task}}', and its subtasks '{{#each subtasks}}{{this}}{{#unless @last}}, {{/unless}}{{/each}}',
    check if these subtasks effectively and comprehensively address the requirements
    of the parent task without any gaps or redundancies, using the following capabilities:
    '{{capabilities_input}}'. Return 'True' if they meet the requirements or 'False' otherwise.
    {{~/user}}
    {{#assistant~}}
    {{gen "result"}}
    {{~/assistant}}''', llm=guidance_gpt4_api)

    response = check_subtasks_program(task=task, subtasks=subtasks, capabilities_input=capabilities_input)
    result = response["result"].strip().lower()

    return result

def get_subtasks(task, state, remaining_decompositions, capabilities_input):
    subtasks_prompt = guidance('''
    {{#system}}You are a helpful agent{{/system}}

    {{#user}}
    Given the task '{{task}}', the current state '{{state}}',
    and {{remaining_decompositions}} decompositions remaining before failing,
    please decompose the task into a detailed step-by-step plan
    that can be achieved using the following capabilities:
    '{{capabilities_input}}'. Provide the subtasks in a comma-separated list,
    each enclosed in square brackets: [subtask1], [subtask2], ...
    {{/user}}
    {{#assistant}}{{gen "subtasks_list"}}{{/assistant}}
    ''', llm=guidance_gpt4_api)

    result = subtasks_prompt(task=task, state=state,
                             remaining_decompositions=remaining_decompositions,
                             capabilities_input=capabilities_input)
    subtasks_with_types = result['subtasks_list'].strip()

    return subtasks_with_types
