import json
import google.generativeai as genai

# Configure Gemini API Key
genai.configure(api_key="AIzaSyCfWqJvycPgVnUrl_OjIN5nzSQxm_iBlw4")

# Load the processed JSON file
def load_processed_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Generate test case prompts for Gemini
def create_test_case_prompt(schema):
    prompt = "Generate structured test cases in JSON format for the given schema:\n\n"
    
    for entity, details in schema.items():
        prompt += f"## Entity: {entity}\nDescription: {details.get('description', '')}\nFields:\n"
        for field, attributes in details["fields"].items():
            prompt += f"- {field} ({attributes['data_type']}): {attributes['description']}\n"
            prompt += f"  Constraints: {', '.join(attributes.get('constraints', []))}\n\n"
    
    prompt += """
### Expected Output JSON Format:
{
    "Rx BC Demographics": [
        {
            "test_case_id": "TC_01",
            "field": "Rx BC Email",
            "test_scenario": "Provide a valid email",
            "expected_result": "Accepted"
        },
        {
            "test_case_id": "TC_02",
            "field": "Rx BC Email",
            "test_scenario": "Leave empty (\\"\\"), Expected: NOT EMPTY",
            "expected_result": "Rejected"
        }
    ]
}
"""
    return prompt

# Call Gemini API
def generate_test_cases(prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    
    # Extract JSON from response (ensure it's valid JSON)
    try:
        # return json.loads(response.text)
        return response.text
    except json.JSONDecodeError:
        print("Error: Gemini response is not valid JSON.")
        return {}

# Save generated test cases to a file
def save_test_cases(test_cases, output_file):
    with open(output_file, "w") as file:
        json.dump(test_cases, file, indent=4)

if __name__ == "__main__":
    processed_json_path = "data/constrains_processed_rules.json"
    output_test_cases_path = "data/generated_test.json"
    
    schema = load_processed_json(processed_json_path)
    prompt = create_test_case_prompt(schema)
    test_cases = generate_test_cases(prompt)
    
    if test_cases:
        save_test_cases(test_cases, output_test_cases_path)
        print(f"Test cases generated and saved to {output_test_cases_path}")
    else:
        print("Failed to generate test cases.")
