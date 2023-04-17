import openai
import os
from dotenv import load_dotenv
from PIL import Image
import requests
from io import BytesIO



load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables or .env file")


# Replace 'your_api_key_here' with your actual API key
openai.api_key = openai_api_key

def get_reply(prompt, model="text-davinci-002", max_tokens=50):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = response.choices[0].text.strip()
    return message


def generate_image(prompt):
        response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
        )
        image_url = response['data'][0]['url']

        return image_url

def try_message():
    input_string = input("Prompt me:")
    reply = get_reply(input_string)
    print(reply)
     
def try_image():
    input_string = input("Prompt me:")
    reply = generate_image(input_string)
    # URL of the generated image
    url = reply

    # Send a GET request to the URL and get the response
    response = requests.get(url)

    # Open the image using Pillow's Image module
    img = Image.open(BytesIO(response.content))

    # Save the image to a directory
    img.save("./images/test_image.jpg")
    img.show()
     
     
def main():
   try_image()
#    try_message()

if __name__ == "__main__":
    main()

