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

def formatImage(path):
    image = Image.open(path)
    width = 1024
    height =1024
    image = image.resize((width, height))
    byte_stream = BytesIO()
    image.save(byte_stream, format='PNG')
    byte_array = byte_stream.getvalue()
    return byte_array
     

def edit_image(prompt):
    response = openai.Image.create_edit(
         
    image=formatImage("./images/static_image.png"),
    mask=formatImage("./images/mask.png"),
    prompt=prompt,
    n=1,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']

    return image_url
def alternatives():
    response = openai.Image.create_variation(
    image=formatImage("./images/test_image.jpg"),
    n=1,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img.show()


def try_message():
    input_string = input("Prompt me:")
    reply = get_reply(input_string)
    print(reply)
     
def try_image():
    input_string = input("Prompt me:")
    to_alt = input("Do you want an alternative too? Y/N ")
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
    if('N' not in to_alt or 'n' not in to_alt):
        img.close()
        alternatives()


def try_edit_image():
    input_string = input("Edit prompt:")
    reply = edit_image(input_string)
    # URL of the generated image
    url = reply
    # Send a GET request to the URL and get the response
    response = requests.get(url)
    # Open the image using Pillow's Image module
    img = Image.open(BytesIO(response.content))
    # Save the image to a directory
    img.save("./images/editted.jpg")
    img.show()
    
     
     
def main():
    #  try_edit_image()
   try_image()
#    try_message()

if __name__ == "__main__":
    main()

