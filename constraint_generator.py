import base64
from openai import OpenAI
import os
import cv2
import json
import parse
import numpy as np
import time
from datetime import datetime

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class ConstraintGenerator:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './vlm_query')
        with open(os.path.join(self.base_dir, 'query.txt'), 'r') as f:
            self.prompt_template = f.read()

    def _build_prompt(self, image_path, instruction):
        img_base64 = encode_image(image_path)
        prompt_text = self.prompt_template.format(instruction=instruction)
        # save prompt
        with open(os.path.join(self.task_dir, 'prompt.txt'), 'w') as f:
            f.write(prompt_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_template.format(instruction=instruction)
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


    def generate(self, instruction='throw pen into the trash bin in front of the robot and table'):
        """
        Args:
            img (np.ndarray): image of the scene (H, W, 3) uint8
            instruction (str): instruction for the query
        Returns:
            save_dir (str): directory where the constraints
        """
        # create a directory for the task
        fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + instruction.lower().replace(" ", "_")
        self.task_dir = os.path.join(self.base_dir, fname)
        os.makedirs(self.task_dir, exist_ok=True)
        # image path
        image_path = os.path.join(self.base_dir, 'query.png')
        # build prompt
        messages = self._build_prompt(image_path, instruction)
        # stream back the response
        stream = self.client.chat.completions.create(model=self.config['model'],
                                                        messages=messages,
                                                        temperature=self.config['temperature'],
                                                        max_tokens=self.config['max_tokens'],
                                                        stream=True)
        output = ""
        start = time.time()
        for chunk in stream:
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
            if chunk.choices[0].delta.content is not None:
                output += chunk.choices[0].delta.content
        print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
        # save raw output
        with open(os.path.join(self.task_dir, 'output_raw.txt'), 'w') as f:
            f.write(output)
        return self.task_dir
