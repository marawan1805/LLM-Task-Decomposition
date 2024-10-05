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
        root_node = TaskNode(self.goal_task)
        previous_task_sets = set()
        task_history = []
        
        while True:
            new_root_node = self.htn_planning_recursive(
                self.initial_state,
                self.goal_task,
                root_node,
                self.max_depth,
                self.capabilities_input,
                db,
                self.send_update_callback,
                task_history
            )
            
            if new_root_node and new_root_node.status == "succeeded":
                print("Plan found successfully!")
                return new_root_node
            
            if new_root_node:
                task_set = frozenset([node.task_name for node in new_root_node.children])
                if task_set in previous_task_sets:
                    print("Repeated tasks detected. Exiting...")
                    return None
                previous_task_sets.add(task_set)
                task_history.extend([node.task_name for node in new_root_node.children])
                root_node = new_root_node
            else:
                break

        print("No valid plan found.")
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
    def decompose(self, task_node, state, depth, max_depth, capabilities_input, goal_state, db, send_update_callback=None,
                task_history=None, n_candidates=3):
        task = task_node.task_name
        decompose_state = state

        if depth > max_depth:
            task_node.status = "failed"
            if send_update_callback:
                send_update_callback(task_node)
            return False, decompose_state

        remaining_decompositions = max_depth - depth
        if remaining_decompositions == 0:
            task_node.status = "failed"
            if send_update_callback:
                send_update_callback(task_node)
            return False, decompose_state

        if is_task_primitive(task):
            translated_task = self.translate_task(task, capabilities_input)
            if can_execute(translated_task, capabilities_input, decompose_state):
                task_node.update_task_name(translated_task)
                print(f"Executing task:\n{translated_task}")
                updated_state = self.execute_task(state, translated_task)
                decompose_state = updated_state
                task_node.status = "completed"
                if send_update_callback:
                    send_update_callback(task_node)
                return True, decompose_state
            else:
                task_node.status = "failed"
                if send_update_callback:
                    send_update_callback(task_node)
                return False, decompose_state

        print(f"Decomposing task:\n{task}")
        
        best_candidate = None
        best_candidate_score = float('-inf')
        candidates = []

        for _ in range(n_candidates):
            subtasks_list = self.get_subtasks(task, decompose_state, remaining_decompositions, capabilities_input, task_history)
            score = self.evaluate_candidate(task, subtasks_list, capabilities_input, task_history)
            candidates.append((subtasks_list, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        success = False
        for subtasks_list, score in candidates:
            if self.check_subtasks(task, subtasks_list, capabilities_input, task_history):
                print(f"Successfully decomposed task into subtasks:\n'{', '.join(subtasks_list)}'")
                success = True
                break

            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = subtasks_list

        if not success and best_candidate is not None:
            print(f"No candidates met the requirements, using the best candidate:\n'{', '.join(best_candidate)}'")
            subtasks_list = best_candidate
            success = True

        if success:
            task_node.status = "in-progress"
            if send_update_callback:
                send_update_callback(task_node)

            for subtask in subtasks_list:
                subtask_node = TaskNode(subtask, parent=task_node)
                task_node.add_child(subtask_node)
                
                success, updated_state = self.decompose(subtask_node, decompose_state, depth + 1, max_depth,
                                                capabilities_input, goal_state, db, send_update_callback, task_history)

                if success:
                    decompose_state = updated_state
                    subtask_node.status = "completed"
                else:
                    subtask_node.status = "failed"
                
                if send_update_callback:
                    send_update_callback(task_node)

            if all(child.status == "completed" for child in task_node.children):
                task_node.status = "completed"
            elif any(child.status == "failed" for child in task_node.children):
                task_node.status = "failed"
            else:
                task_node.status = "in-progress"

            if send_update_callback:
                send_update_callback(task_node)

            if task_node.status == "completed":
                db.add_task_node(task_node)

            return task_node.status == "completed", decompose_state
        else:
            task_node.status = "failed"
            if send_update_callback:
                send_update_callback(task_node)
            return False, decompose_state

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
    def get_subtasks(self, task, state, remaining_decompositions, capabilities_input, task_history):
        subtasks_with_types = get_subtasks(task, state, remaining_decompositions, capabilities_input, task_history)
        print(f"Decomposing task {task} into candidates:\n{subtasks_with_types}")
        subtasks_list = extract_lists(subtasks_with_types)
        return subtasks_list


    @trace_function_calls
    def execute_task(self, state, task):
        prompt = (f"Given the current state '{state}' and the task '{task}', "
                f"update the state after executing the task:")

        response = call_groq_api(prompt)

        updated_state = response.choices[0].message.content.strip()
        log_response("execute_task", task)
        log_state_change(state, updated_state, task)
        return updated_state
