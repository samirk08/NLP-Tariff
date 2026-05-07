from flask import Flask, render_template, request, redirect, url_for
import os
import base64
import base64
import requests
import os
from UROP_matching import get_hs_code
from openai import OpenAI
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Other routes remain unchanged...

api_key = "REDACTED_OPENAI_KEY"    # Replace with your actual OpenAI API key

# Set up environment variables
def setup_environment(api_key):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OPENAI_API_KEY"] = api_key

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to get image description from OpenAI
def get_image_description(api_key, image_path):
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please provide a concise description of the primary item in this image, focusing on its identifiable and classifiable features relevant for customs and tariff description purposes. Use no more than 10 words. Only describe the primary item in the image and not the entire scene."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()
    if 'choices' in response_data:
        return response_data['choices'][0]['message']['content']
    else:
        raise Exception(response_data['error']['message'])

# Function to get HS code
def get_hs_code_with_retries(description, retries=3):
    for i in range(retries):
        try:
            hs_code, product = get_hs_code(description, headless=False)
            return hs_code, product
        except Exception as e:
            if i == retries - 1:
                raise e
            continue


app = Flask(__name__)

# If you want to keep your API key in an environment variable:
# (On your system: export OPENAI_API_KEY=your_key)
# then read it like this:
# api_key = os.environ.get("OPENAI_API_KEY", "fallback_api_key")

# A route for the homepage
@app.route('/')
def index():
    # This renders the index.html template
    return render_template('index.html')

# 1) Process a text description
@app.route('/process_description', methods=['POST'])
def process_description():
    description = request.form.get('description')
    
    # ---- Use your existing code here ----
    # e.g., call the get_hs_code_with_retries or something similar:
    try:
        hs_code, product = get_hs_code_with_retries(description)
        # result_text = f"Selected HS code: {hs_code}, Product: {product}"
        result_text = f"Product: {product}"
    except Exception as e:
        result_text = f"Error getting HS code: {str(e)}"

    # Option 1: Just display the result in the browser (quick approach)
    # return result_text

    # Option 2: Render the template again with the result
    return render_template('index.html', result=result_text)

# 2) Process an uploaded image
@app.route('/process_image', methods=['POST'])
def process_image():
    # Grab the file from the request
    image_file = request.files.get('image_file')

    if not image_file:
        return "No image uploaded."

    # Save the file to a temporary location (optional)
    # or convert it to base64 in-memory, etc.

    # Example: save to disk in a "uploads" folder
    upload_path = os.path.join('uploads', image_file.filename)
    image_file.save(upload_path)

    # Now feed the image into your existing get_image_description / get_hs_code code
    try:
        # Step 1: get the GPT-based description
        gpt_description = get_image_description(api_key, upload_path)

        # Step 2: pass that description to get_hs_code
        hs_code, product = get_hs_code_with_retries(gpt_description)
        
        result_text = (
            f"Image description: {gpt_description}\n"
            # f"Selected HS code: {hs_code}, Product: {product}"
            f"Product: {product}"
        )
    except Exception as e:
        result_text = f"Error processing image: {str(e)}"

    # Option to return plain text or re-render the template with a result
    return render_template('index.html', result=result_text)

if __name__ == '__main__':
    # Make sure to create an 'uploads' folder, or handle that logic differently
    if not os.path.exists('uploads'):
        os.mkdir('uploads')

    app.run(debug=True)
