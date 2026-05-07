from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from openai import OpenAI
import time
import os
from selenium.webdriver.chrome.options import Options


os.environ["TOKENIZERS_PARALLELISM"] = "false"
if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError("Set the OPENAI_API_KEY environment variable before running this script.")
client = OpenAI()

##### GPT Functions #####
# General GPT call function
def ask_gpt(prompt, system_prompt, model_name="gpt-4"):
    """
    Queries GPT model to enhance descriptions or generate content based on prompts.
    
    Args:
        prompt (str): The user prompt to send to the model.
        system_prompt (str): Instructions for the model on how to handle the user prompt.
        model_name (str): Specifies the model version.
    
    Returns:
        str: The model's response text.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# prompt for picking an option
def create_gpt_prompt(product_description, options):
    prompt = f"""
You are an assistant trained to classify products based on their descriptions and available options.

Product Description: "{product_description}"

Here are the available options:
"""
    for i, option in enumerate(options, 1):
        prompt += f"{i}) {option}\n"
    
    prompt += """
Can you pick which one of the options best describes the product? ONLY PICK ONE OF THE OPTIONS from the options given. Even if the product description does not provide information about the question, please select the most appropriate option based on the information provided.

IMPORTANT:
- Only reply with the exact text of the selected option from the list above.
- Do not include any additional text, explanations, or numbering.
- INCLUDE ONE OF THE OPTIONS FROM THE LIST ABOVE.
"""
    return prompt

# function to create GPT prompt to fill out composition table
def create_gpt_composition_prompt(product_description, composition_options):
    # Start building the prompt
    composition_prompt = (
        "You are an assistant trained to determine the approximate composition of "
        "the product in terms of percentages for the following categories:\n\n"
    )
    
    # List out each category
    for option in composition_options:
        composition_prompt += f"- {option}\n"
    
    # Continue with instructions
    composition_prompt += f"""
The product description is:""" + f'"{product_description}"' + """

INSTRUCTIONS:
1. The sum of all percentages across the categories must exactly equal 100.
2. You must respond ONLY with a single Python-style list of numeric values 
   (ints or floats), in the same order as the categories listed above.
3. Do not include any extra text or explanation.

For example, if you decide that:
- {composition_options[0]}: 10%
- {composition_options[1]}: 20%
- {composition_options[2]}: 70%

You would output:

[10, 20, 70]
"""
    return composition_prompt

def gpt_response_valid(response, options):
    for option in options:
        if option.lower() in response.lower():
            return True
    return False

def modernize_product_description(product_description):
    prompt = f"""
You are an assistant trained to modernize older tariff product descriptions to meet current classification standards. Maintain the original product description while enhancing it with more detailed and specific information suitable for accurate classification. 

- Original Product Description: "{product_description}"

Your output should follow this format:
"{product_description}: [additional specifics and detailed description]"

Focus on clarity, precision, and relevance to tariff standards in your additions.
"""
    return prompt

###########################


#########################################

# prompt for picking an HS code
def create_hs_code_prompt(product_description, hs_options):
    prompt = f"""
You are an assistant trained to classify products based on their descriptions and available HS code options.

Product Description: "{product_description}"

Here are the available HS code options:
"""
    for option in hs_options:
        prompt += f"- {option}\n"
    
    prompt += """
IMPORTANT:
- Based on the product description, which HS code is the most appropriate? 
- When the product description is not very descriptive, choose the most general but specific classification (the fewest digits) that matches the description without making unnecessary assumptions.
- Only reply with the HS code (e.g., 64.05 or 6405.90) and nothing else.
"""
    return prompt

def handle_dropdown(driver, wait, dropdown_css_selector, option_text):
    # Wait for the dropdown to be visible
    try:
        dropdown = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, dropdown_css_selector)))
    except:
        print(f"Dropdown with selector '{dropdown_css_selector}' not visible.")
        return False

    # Get the options within the dropdown
    options = dropdown.find_elements(By.TAG_NAME, "li")
    for option in options:
        if option_text.lower() in option.text.lower():
            option.click()  # Click on the matching option
            print(f"Clicked on option: {option.text}")
            return True

    print(f"Option '{option_text}' not found in the dropdown.")
    return False

def click_most_relevant_option(driver, product_description, system_prompt):
    """
    Function to click the most relevant option based on the product description.
    If the website then asks for a composition table, we fill it automatically.

    Args:
        driver (webdriver): The webdriver instance.
        product_description (str): The product description to classify.
        system_prompt (str): Instructions for the model on how to handle the user prompt. 
    """
    try:
        list_items = driver.find_elements(By.TAG_NAME, "li")
        options = [li.text for li in list_items]
        filtered_options = find_filtered_options(options)
        filtered_list_items = [li for li in list_items if li.text.strip() in filtered_options]
        
        # breakpoint()
        if 'Accept' in filtered_options:
            # this means that we need to fill out the compositions table
            # STEPS: 1. Run GPT call to make a prompt to fill out the table
            #       2. Fill out the table in accordance to the website
            #       3. Click on the 'Accept' button
            # make a list called composition_options that goes up to "Accept"
            composition_options = filtered_options[:filtered_options.index('Accept')]
            # make a prompt for GPT to fill out the table
            prompt = create_gpt_composition_prompt(product_description, composition_options)
            gpt_response = ask_gpt(prompt, system_prompt)
            # convert the response to a list of numbers. the gpt_response is a string of a list of numbers
            response_list_str = gpt_response.strip("[] \n\r")
            response_list = [val.strip() for val in response_list_str.split(",")]
            # now we need to fill out the table
            # get the table
            # table = driver.find_element(By.CSS_SELECTOR, "table.ccce-table.ccce-table-fixed")
            table_ul = driver.find_element(By.CSS_SELECTOR, "ul.ccce-dropdown-options.ccce-show")
            composition_items = table_ul.find_elements(By.TAG_NAME, "li")

            # 5) Fill out each composition item with the corresponding percentage
                #    from GPT's response. 
                #    For example, if composition_items correspond to:
                #        - Cotton
                #        - Polyester
                #        - Other
                #    and response_list is [10, 20, 70], then we fill 10 for Cotton, 20 for Polyester, etc.
            for i, item in enumerate(composition_items):
                # Safety check to avoid index errors
                if i < len(response_list):
                    # Example: find an <input> inside the li
                    try:
                        input_el = item.find_element(By.TAG_NAME, "input")
                        input_el.clear()
                        input_el.send_keys(response_list[i])
                        print(f"Filled '{composition_options[i]}' with {response_list[i]}%")
                    except Exception as e:
                        print(f"Could not fill composition item #{i}: {e}")

                # 6) Finally, click on the 'Accept' item within the same list (or a separate button)
            try:
                accept_item = [li for li in composition_items if li.text.strip() == 'Accept']
                if accept_item:
                    accept_item[0].click()
                    print("Clicked on 'Accept' button/option.")
                else:
                    print("Could not find 'Accept' item to click.")
            except Exception as e:
                print(f"Error clicking 'Accept': {e}")
            return True

        if not filtered_list_items:
            print("No questions asked.")
            return False  # Exit if no options

        # Modernize the product description
        ask_gpt_product_description = modernize_product_description(product_description)
        gpt_pd = ask_gpt(ask_gpt_product_description, system_prompt)

        # Prompt GPT for the best option
        prompt = create_gpt_prompt(gpt_pd, filtered_options)
        gpt_response = ask_gpt(prompt, system_prompt)

        # If GPT's pick doesn't match any option, pick the last option as a fallback
        if not gpt_response_valid(gpt_response, filtered_options):
            filtered_list_items[-1].click()
            print(f"GPT response invalid. Picked fallback: {filtered_list_items[-1].text}")
        else:
            # Try to click exactly GPT's matching option
            clicked_option = False
            for li in list_items:
                if li.text.strip() == gpt_response.strip():
                    li.click()
                    print(f"Clicked option: {li.text}")
                    clicked_option = True
                    break
            # If not found for some reason, pick fallback
            if not clicked_option:
                print(f"Option '{gpt_response}' not found among list items. Picking fallback.")
                filtered_list_items[-1].click()

        # breakpoint()
        # --- New: Attempt to detect composition table and fill it ---
        # Wait a moment in case the composition table loads after clicking.
        time.sleep(1)

        # Run the detect/fill function (this will no-op if not found)
        wait = WebDriverWait(driver, 10)

        return True

    except Exception as e:
        print(f"Error in click_most_relevant_option: {e}")
        return False


def wait_for_dropdown(wait, dropdown_css_selector):
    try:
        return wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, dropdown_css_selector)))
    except Exception as e:
        print(f"Dropdown not found: {e}")
        return None
    
def find_filtered_options(options):
    stop_item = "Appraise trading trends and market outlook"
    # Filter out empty strings and stop at the specified item
    filtered_options = []
    for item in options:
        if item.strip() == stop_item:
            break
        if item.strip():  # Only include non-empty strings
            filtered_options.append(item.strip())
    return filtered_options

def get_hs_code(product_description, headless=True):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    if headless:
        driver = webdriver.Chrome(options=chrome_options) # uncomment for headless
    else:
        driver = webdriver.Chrome()  
        
    driver.get("https://uscensus.prod.3ceonline.com")

    # Wait for the page to load and locate the input field
    wait = WebDriverWait(driver, 20)
    input_box = wait.until(EC.visibility_of_element_located((By.ID, "ccce-queryBox")))

    # Enter product description (e.g. "shoes")
    input_box.send_keys(product_description)

    # Click the "Classify" button
    classify_btn = driver.find_element(By.XPATH, "//a[contains(text(), 'Classify')]")
    classify_btn.click()

    system_prompt = "Assist in selecting the most relevant classification option or the most appropriate composition percentages based on the given product description and question."

    time.sleep(2)
    while True:
        # breakpoint()
        success = click_most_relevant_option(driver, product_description, system_prompt)
        if success:
            # If the function returned True, wait some time and try again
            time.sleep(2)
        else:
            # No more suitable options or no options found, move on
            break    ### PICK THE MOST BEST HS OPTION
    
    table = driver.find_element(By.CSS_SELECTOR, "table.ccce-table.ccce-table-fixed")

    # Locate the rows within the table's body
    rows = table.find_element(By.TAG_NAME, "tbody").find_elements(By.TAG_NAME, "tr")

    # Extract the HS codes and descriptions
    hs_options = []
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")  # Get all cells in the row
        if len(cells) >= 2:  # Ensure there are at least two cells (code and description)
            hs_code = cells[0].text.strip()  # First cell: HS code
            description = cells[1].text.strip()  # Second cell: Description
            hs_options.append(f"{hs_code}: {description}")

    # given the HS options, ask GPT to pick the best one
    prompt = create_hs_code_prompt(product_description, hs_options)
    gpt_response = ask_gpt(prompt, system_prompt)

    selected_hs_code = gpt_response.strip()
    selected_product = [option for option in hs_options if selected_hs_code in option][0]

    return selected_hs_code, selected_product

def main():
    # product_description = "distilled spirits of Jamaica proof"
    product_description = 'distilled spirits of Jamaica proof' #website edge case
    
    selected_hs_code, selected_product = get_hs_code(product_description, headless=False)

    print(f"Selected HS code: {selected_hs_code}, Product: {selected_product}")   

if __name__ == "__main__":
    main()