import os
import openai
import tiktoken
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

client = openai.OpenAI()


def get_completion(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """Get a simple chat completion from a model through the OpenAI API"""
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


def get_completion_from_messages(messages: list[dict[str, str]],
                                 model: str = "gpt-3.5-turbo",
                                 temperature: float = 0,
                                 max_tokens: int = 500) -> str:
    """Get a chat completion from a model through the OpenAI API given a list of messages, temperature and max_tokens
    in the model's response"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def get_completion_and_token_count(messages: list[dict[str, str]],
                                   model: str = "gpt-3.5-turbo",
                                   temperature: float = 0,
                                   max_tokens: int = 500) -> [str, dict]:
    """Get a chat completion from a model through the OpenAI API given a list of messages, temperature and max_tokens
    in the model's response and also, get a token count for the request"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    content = response.choices[0].message.content

    token_dict = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens
    }

    return content, token_dict


def main() -> None:
    messages = [
        {'role': 'system',
         'content': """You are an assistant who responds\
     in the style of Dr Seuss."""},
        {'role': 'user',
         'content': """write me a very short poem \ 
     about a happy carrot"""},
    ]
    response, token_dict = get_completion_and_token_count(messages)
    print(response)
    print(token_dict)


if __name__ == '__main__':
    main()
