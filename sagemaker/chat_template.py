from transformers import AutoTokenizer

model_name = "lightonai/alfred-40b-1023"

template = """{% for message in messages %}{% if message['role'] == 'user' %}{{ '<start_user>' + message['content'].strip() + '<end_message>' }}{% elif message['role'] == 'system' %}{{ '<start_system>' + message['content'].strip() + '<end_message>' }}{% elif message['role'] == 'assistant' %}{{ '<start_assistant>'  + message['content'] + '<end_message>' }}{% else %}{{ raise_exception('Only system, user and assistant roles are supported.') }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<start_assistant>' }}{% endif %}{% endfor %}"""

tokenizer = AutoTokenizer.from_pretrained(model_name)

# only uncomment if you want to push the template to the hub
# tokenizer.chat_template = template
# tokenizer.push_to_hub(model_name)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "Hello, how are you?"
    },
    {
        "role": "assistant",
        "content": "I am fine, thanks!"
    },
    {
        "role": "user",
        "content": "What is your name?"
    },
]

prompt = tokenizer.apply_chat_template(messages,
                                       tokenize=False,
                                       add_generation_prompt=True)
print(prompt)
