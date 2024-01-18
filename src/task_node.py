import uuid

from text_utils import trace_function_calls


class TaskNode:
    def __init__(self, task_name, parent=None, status="pending"):
        self.task_name = task_name
        self.node_name = str(uuid.uuid4())
        self.parent = parent
        self.children = []
        self.status = status

    @trace_function_calls
    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

    @trace_function_calls
    def remove_child(self, child_node):
        if child_node in self.children:
            self.children.remove(child_node)
            child_node.parent = None

    @trace_function_calls
    def update_task_name(self, task_name):
        self.task_name = task_name

    def all_children_succeeded(self):
        return all(child.status == 'succeeded' for child in self.children)

    def mark_as_succeeded(self):
        if self.all_children_succeeded():
            self.status = 'succeeded'
        else:
            print(f"Cannot mark {self.task_name} as succeeded because some children tasks are still pending.")