import openai
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential

openai.api_key = os.environ.get("OPENAI_API_KEY")


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_with_backoff(**kwargs):
    """Backoff to combat with rate limits"""
    if kwargs.get("session_id") is None:
        response = openai.ChatCompletion.create(model=kwargs["model"], messages=kwargs["messages"])
    else:
        response = openai.ChatCompletion.create(
            model=kwargs["model"],
            messages=kwargs["messages"],
            session_id=kwargs.get("session_id"),
        )
    return response
