import json
import os
import google.generativeai as genai
import pandas as pd
import yaml
import re

def load_config(config_path="config/settings.yaml"):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        return None

def clean_json_response(response_text):
    """
    Cleans and parses a JSON response from the LLM.
    This function first attempts to parse the raw response.
    If that fails, it applies a series of cleaning steps to correct common formatting errors.
    """
    print("\nRaw Response from LLM:\n" + "-" * 80 + "\n" + response_text + "\n" + "-" * 80)

    # Attempt 1: Try to parse the raw response
    try:
        print("Attempting to parse raw JSON...")
        return json.loads(response_text)
    except json.JSONDecodeError as e1:
        print(f"Attempt 1 failed: {e1}")

        # Attempt 2: Clean and parse
        try:
            print("Attempting to clean and parse...")

            # Remove code block markers
            response_text = response_text.replace("```json", "").replace("```", "").strip()

            # Remove trailing commas within objects
            response_text = re.sub(r',\s*}', '}', response_text)

            # Remove trailing commas within arrays
            response_text = re.sub(r',\s*]', ']', response_text)

            # Fix unescaped characters
            response_text = response_text.replace('\\"', '"')

            print("\nCleaned JSON:\n" + "-" * 80 + "\n" + response_text + "\n" + "-" * 80)
            cleaned_json = json.loads(response_text)
            return cleaned_json

        except json.JSONDecodeError as e2:
            print(f"Attempt 2 failed: {e2}\nCould not parse JSON after cleaning. Please check the LLM output.")
            return None

def generate_test_cases(field_name, data_type, constraints, llm_client, llm_model, max_output_tokens=1000):
    """Generates test cases using an LLM (Gemini or OpenAI)."""
    prompt = f"""
    Generate comprehensive test cases for a field named '{field_name}' with data type '{data_type}'.
    The field has the following constraints: {constraints}.
    Include valid cases, invalid cases, edge cases, and boundary conditions.
    Format the output as a JSON list where each item contains these fields: "test_case", "description", "expected_result", "input".
    IMPORTANT: Return valid JSON ONLY.
    Example Output:
    [
      {{"test_case": "Valid Input", "description": "Basic valid input test", "expected_result": "Pass", "input": "test"}},
      {{"test_case": "Invalid Input", "description": "Basic invalid input test", "expected_result": "Fail", "input": null}}
    ]
    """

    try:
        if isinstance(llm_client, genai.GenerativeModel):
            print(f"Generating test cases for {field_name}...")
            print("Sending prompt to LLM:")
            # Print prompt here, use logging for production
            # print("-" * 80 + "\n" + prompt + "\n" + "-" * 80)

            response = llm_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_output_tokens
                )
            )
            if hasattr(response, 'text'):
                test_cases = clean_json_response(response.text)
                if test_cases is not None:
                     # Ensure "expected_result" is either "Pass" or "Fail"
                    for case in test_cases:
                        if 'expected_result' in case:
                            case['expected_result'] = case['expected_result'].capitalize()
                        else:
                            print(f"Warning: No 'expected_result' in test case for {field_name}")

                    return test_cases
                else:
                    print(f"Failed to generate or parse test cases for {field_name}")
                    return None
            else:
                print(f"Error: LLM Response missing 'text' attribute.")
                return None

        else:
            raise ValueError("Unsupported LLM client type.")

    except Exception as e:
        print(f"Exception in generate_test_cases: {e}")
        return None

def generate_test_cases_from_rules(rules, llm_client, llm_model, config):
    all_test_cases = {}
    for parent_field, details in rules.items():
        print(f"Processing parent field: {parent_field}...")
        for field_name, field_details in details["fields"].items():
            print(f"Generating test cases for: {field_name}...")
            test_cases = generate_test_cases(
                field_name,
                field_details["data_type"],
                field_details["constraints"],
                llm_client,
                llm_model
            )

            if test_cases:
                all_test_cases[f"{parent_field}.{field_name}"] = test_cases
                print(f"Successfully generated {len(test_cases)} test cases for {field_name}")
            else:
                print(f"Failed to generate test cases for {field_name}")

    return all_test_cases

def save_test_cases(all_test_cases, output_file):
    """Saves the generated test cases to a JSON file."""
    try:
        with open(output_file, "w") as f:
            json.dump(all_test_cases, f, indent=4)
        print(f"✅ Test cases saved to {output_file}")
    except IOError as e:
        print(f"Error saving test cases to {output_file}: {e}")

def generate_test_cases_from_file(config):
    """Generates test cases based on rules in a JSON file."""
    # Initialize LLM
    if config.get("gemini_api_key"):
        print("Configuring Gemini...")
        genai.configure(api_key=config["gemini_api_key"])
        llm_model = config.get("gemini_model", "gemini-1.5-flash")
        llm_client = genai.GenerativeModel(llm_model)
        print(f"Using Gemini model: {llm_model}")
    else:
        print("Error: No LLM API key found in config.yaml")
        return

    # Load rules
    rules_file = config.get("constrains_processed_rules_file")
    if not rules_file:
        print("Error: constrains_processed_rules_file not found in config")
        return

    try:
        with open(rules_file, "r") as f:
            rules = json.load(f)
            print(f"Successfully loaded rules from {rules_file}")
    except FileNotFoundError:
        print(f"Error: Rules file not found at {rules_file}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {rules_file}: {e}")
        return

    all_test_cases = generate_test_cases_from_rules(rules, llm_client, llm_model, config)

    # Save results
    output_file = config.get("generated_test_cases_file")
    if not output_file:
        print("Error: generated_test_cases_file not found in config")
        return

    save_test_cases(all_test_cases, output_file)

if __name__ == "__main__":
    config = load_config()
    if config is None:
        exit()

    generate_test_cases_from_file(config)