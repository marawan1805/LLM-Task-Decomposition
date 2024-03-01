
The HTN Planner leverages OpenAI's GPT and Hierarchical Task Network (HTN) principles to autonomously create comprehensive plans. It initiates with broad objectives, breaking them down into manageable subtasks through the Large Language Model (LLM), refining these until they are actionable.

Optimal performance is achieved with GPT-4, though adaptation of other open-source LLMs is feasible with necessary adjustments.

Key Features:

Decomposition: Analyzes a task, breaking it into finer subtasks until reaching a preset depth or encountering a planning obstacle, with mechanisms to select the most viable decomposition route, potentially concluding prematurely if satisfactory outcomes are attained.
Re-planning: Engages when initial plans falter or face disruptions, to formulate alternative strategies.
Task Execution: Defines tasks as entities ready for implementation, noting current versions do not facilitate actual execution within a terminal.
State Monitoring: Updates and monitors the state throughout the task execution phase.
Text Interpretation: Extracts actionable data from LLM-generated textual responses.
Task Conversion: Transforms basic tasks into executable commands or code.
User Interface: Incorporates a straightforward React interface to visually represent planning structures.
Logging: Generates extensive logs in designated directories for system analysis, including detailed tracking of function calls and subsystem operations, along with error identification and state transition records.
Future Directions:

Archive successful plan components in a vector database to cut down on future planning resources.
Enhance text interpretation capabilities to address broader scenarios.
Introduce additional post-processing steps.
Reassess task execution prerequisites.
Setup Instructions:

Backend:

Assign your OpenAI API key to the OPENAI_KEY environment variable.
Install necessary libraries via pip install -r requirements.txt.
Initiate the planning tool with python src/main.py, inputting initial conditions and goals, along with specifying default tools and selecting the planning algorithm of choice.
Frontend:

Navigate to the frontend folder (cd src/frontend) and launch it (npm start).

Credits:
Chatgpt
Daemonib
