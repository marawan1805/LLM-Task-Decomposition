import threading

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from htn_planner import HTNPlanner
# from search_planner import SearchPlanner

from LLM_utils import get_initial_task, compress_capabilities
from text_utils import trace_function_calls

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@trace_function_calls
def task_node_to_dict(task_node):
    if task_node is None:
        return None

    return {
        "task_name": task_node.task_name,
        "status": task_node.status,
        "children": [task_node_to_dict(child) for child in task_node.children]
    }

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def send_task_node_update(task_node):
    root_task_node = task_node
    while root_task_node.parent is not None:
        root_task_node = root_task_node.parent
    task_node_data = task_node_to_dict(root_task_node)
    socketio.emit('task_node_update', task_node_data)

def task_node_to_dict(task_node):
    return {
        "task_name": task_node.task_name,
        "status": task_node.status,
        "children": [task_node_to_dict(child) for child in task_node.children]
    }

def run_server():
    socketio.run(app, host="127.0.0.1", debug=True, use_reloader=False, port=5000, allow_unsafe_werkzeug=True, log_output=False)

def print_plan(task_node, depth=0):
    print(f"{'  ' * depth}- {task_node.task_name}")
    for child in task_node.children:
        print_plan(child, depth + 1)

def main():
    initial_state = input("Describe the initial state: ")
    goal = input("Describe your goal: ")
    default_capabilities = "Manipulation actions (grab, push, pull, ...); Movement actions (move, reach, ...); Kitchen tasks (cook, bake, boil, ...); Cleaning tasks (clean, wipe, vacuum, ...); Miscellaneous tasks (scan, activate, identify, ...)"
    print(f"Default capabilities: {default_capabilities}")
    capabilities_input = input("Describe the capabilities available (press Enter to use default): ")
    if not capabilities_input:
        capabilities_input = default_capabilities

    compressed_capabilities = compress_capabilities(capabilities_input)
    goal_task = get_initial_task(goal)

    print("\nUsing default HTN planner")
    print("Starting server...")

    htn_planner = HTNPlanner(goal, initial_state, goal_task, compressed_capabilities, send_update_callback=send_task_node_update)
    
    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    plan = htn_planner.htn_planning()

    if plan:
        print("\nFinal plan:")
        print_plan(plan)
    else:
        print("\nNo valid plan found.")

    server_thread.join()

if __name__ == '__main__':
    main()