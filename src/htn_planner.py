# An implementation of HTN using a LLM API
# Due to the expressiveness of language, a lot of steps that would generally require complex functions are left up
# to the LLM

from LLM_utils import groq_is_goal, is_task_primitive, can_execute, log_state_change
from LLM_api import call_groq_api, log_response
from task_node import TaskNode
from text_utils import extract_lists, trace_function_calls
from htn_prompts import *
from vector_db import VectorDB

class HTNPlanner:
    def __init__(self, goal_input, initial_state, goal_task, capabilities_input, max_depth=5, send_update_callback=None):
        self.goal_input = goal_input
        self.initial_state = initial_state
        self.goal_task = goal_task
        self.capabilities_input = capabilities_input
        self.max_depth = max_depth
        self.send_update_callback = send_update_callback

    def htn_planning(self):
        db = VectorDB()
        root_node = TaskNode(self.goal_input)
        max_iterations = 100  # Adjust this value as needed

        print(f"Initial goal: {self.goal_task}")
        success, _ = self.decompose(root_node, self.initial_state, 0, self.max_depth, 
                                    self.capabilities_input, self.goal_task, db, self.send_update_callback)
        
        if success and root_node.status == "completed":
            print("Plan found successfully!")
            return root_node
        else:
            print("Failed to find a valid plan.")
            return None

    @trace_function_calls
    def htn_planning_recursive(self, state, goal_task, root_node, max_depth, capabilities_input, db, send_update_callback=None, task_history=None):
        if groq_is_goal(state, goal_task):
            return root_node

        if send_update_callback:
            send_update_callback(root_node)

        success, updated_state = self.decompose(root_node, state, 0, max_depth, capabilities_input, goal_task,
                                        db, send_update_callback, task_history)
        if success:
            root_node.status = "succeeded"
            state = updated_state
            return root_node
        else:
            root_node.status = "failed"

        return root_node

    @trace_function_calls
    def replan_required(self, state, goal_task, task_node):
        if groq_is_goal(state, goal_task):
            return False
        if task_node is None or task_node.children == []:
            return True
        return False


    @trace_function_calls
    def translate_task(self, task, capabilities_input):
        response = translate(self.goal_input, task, capabilities_input)
        translated_task = response.strip()
        log_response("translate_task", translated_task)
        return translated_task


    @trace_function_calls
    def check_subtasks(self, task, subtasks, capabilities_input, task_history):
        result = check_subtasks(task, subtasks, capabilities_input, task_history)
        log_response("check_subtasks", result)
        return result == 'true'

    @trace_function_calls
    def decompose(self, task_node, state, depth, max_depth, capabilities_input, goal_state, db, send_update_callback=None, task_history=None):
        task = task_node.task_name
        decompose_state = state

        print(f"Decomposing task (depth {depth}/{max_depth}): {task}")

        if depth > max_depth:
            print(f"Max depth reached for task: {task}")
            task_node.status = "failed"
            if send_update_callback:
                send_update_callback(task_node)
            return False, decompose_state

        subtasks_list = self.get_subtasks(task, decompose_state, max_depth - depth, capabilities_input, task_history)
        print(f"Subtasks for {task}: {subtasks_list}")

        if not subtasks_list:
            print(f"No valid subtasks found for {task}")
            task_node.status = "failed"
            if send_update_callback:
                send_update_callback(task_node)
            return False, decompose_state

        task_node.status = "in-progress"
        if send_update_callback:
            send_update_callback(task_node)

        for subtask in subtasks_list:
            subtask_node = TaskNode(subtask, parent=task_node)
            task_node.add_child(subtask_node)
            
            if is_task_primitive(subtask):
                if can_execute(subtask, capabilities_input, decompose_state):
                    print(f"Executing task: {subtask}")
                    updated_state = self.execute_task(decompose_state, subtask)
                    decompose_state = updated_state
                    subtask_node.status = "completed"
            else:
                success, updated_state = self.decompose(subtask_node, decompose_state, depth + 1, max_depth,
                                                capabilities_input, goal_state, db, send_update_callback, task_history)
                if success:
                    decompose_state = updated_state
                else:
                    subtask_node.status = "failed"

            if send_update_callback:
                send_update_callback(subtask_node)

            if subtask_node.status == "failed":
                task_node.status = "failed"
                if send_update_callback:
                    send_update_callback(task_node)
                return False, decompose_state

        task_node.status = "completed"
        if send_update_callback:
            send_update_callback(task_node)

        if task_node.status == "completed":
            db.add_task_node(task_node)

        print(f"Task completed: {task}")
        return True, decompose_state

    @trace_function_calls
    def evaluate_candidate(self, task, subtasks, capabilities_input, task_history):
        max_retries = 3
        retries = 0
        while retries < max_retries:
            # Max 10 token or 8 digits after the decimal 0.99999999
            response = evaluate_candidate(self.goal_input, task, subtasks, capabilities_input, task_history)
            try:
                score = float(response.strip())
                log_response("evaluate_candidate", score)
                return score
            except ValueError:
                retries += 1
                if retries >= max_retries:
                    raise ValueError("Failed to convert response to float after multiple retries.")


    @trace_function_calls
    def get_subtasks(self, task, state, remaining_decompositions, capabilities_input, task_history=None):
        subtasks_with_types = get_subtasks(task, state, remaining_decompositions, capabilities_input, task_history or [])
        print(f"Decomposing task {task} into candidates:\n{subtasks_with_types}")
        return subtasks_with_types


    @trace_function_calls
    def execute_task(self, state, task):
        prompt = (f"Given the current state '{state}' and the task '{task}', "
                f"update the state after executing the task:")

        response = call_groq_api(prompt)

        updated_state = response.choices[0].message.content.strip()
        log_response("execute_task", task)
        log_state_change(state, updated_state, task)
        return updated_state
