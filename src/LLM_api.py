import os
import time
import datetime
from groq import Groq
from ratelimit import limits, sleep_and_retry

# Define the rate limit (10 calls per minute)
CALLS = 10
RATE_LIMIT = 60

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def call_groq_api(prompt, max_tokens=None, temperature=1.0, strip=False):
    retries = 3
    delay = 5

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    while retries > 0:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-70b-8192",
                max_tokens=max_tokens,
                temperature=temperature
            )
            return chat_completion.choices[0].message.content.strip() if strip else chat_completion
        except Exception as e:
            print(f"Error encountered: {e}. Retrying in {delay} seconds...")
            retries -= 1
            time.sleep(delay)

    raise Exception("Failed to get a response from Groq API after multiple retries.")

updated_log_files = {}


def log_response(function_name, response):
    global updated_log_files

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = f"{log_dir}/{function_name}.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file_path, "a") as log_file:
        if function_name not in updated_log_files:
            log_file.write("\n--- Application run start ---\n")
            updated_log_files[function_name] = True
        log_file.write(f"{timestamp}:\n{response}\n")