import openai

"""
Prompt: GPT-o1
"""
def message_gpt_o1(prompt_template, instruction):
    messages = [
        # User Role
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_template.format(instruction=instruction)
                }
            ]
        }
    ]
    return messages

"""
Prompt: GPT-o-latest
"""
def message_gpt_o(prompt_template, instruction, img_base64):
    messages = [
        # System Role
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": prompt_template.format(instruction=instruction)
                }
            ]
        },
        # User Role
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": instruction
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                },
            ]
        }
    ]
    return messages