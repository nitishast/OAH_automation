import json
import os
from typing import Dict, List, Optional, Any
import google.generativeai as genai
import yaml
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_generation.log'),
        logging.StreamHandler()
    ]
)

class TestCaseGenerator:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.llm_client = self._initialize_llm()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file with error handling."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load config: {str(e)}")
            raise

    def _initialize_llm(self) -> genai.GenerativeModel:
        """Initialize the LLM client with error handling."""
        try:
            if not self.config.get("gemini_api_key"):
                raise ValueError("Gemini API key not found in config")
            
            genai.configure(api_key=self.config["gemini_api_key"])
            model_name = self.config.get("gemini_model", "gemini-1.5-flash")
            return genai.GenerativeModel(model_name)
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {str(e)}")
            raise

    def _generate_prompt(self, field_name: str, data_type: str, constraints: List[str]) -> str:
        """Generate a more structured and specific prompt for test case generation."""
        return f"""
Generate test cases for the field '{field_name}' with following specifications:
- Data Type: {data_type}
- Constraints: {constraints}

Requirements:
1. Include ONLY the JSON array of test cases in your response
2. Each test case must have these exact fields:
   - "test_case": A clear, unique identifier for the test
   - "description": Detailed explanation of what the test verifies
   - "expected_result": MUST be exactly "Pass" or "Fail"
   - "input": The test input value (can be null, string, number, etc.)

3. Include these types of test cases:
   - Basic valid inputs
   - Basic invalid inputs
   - Null/empty handling
   - Boundary conditions
   - Edge cases
   - Type validation

Return the response in this exact format:
[
    {{
        "test_case": "TC001_Valid_Basic",
        "description": "Basic valid input test",
        "expected_result": "Pass",
        "input": "example"
    }},
    {{
        "test_case": "TC002_Invalid_Null",
        "description": "Test with null input",
        "expected_result": "Fail",
        "input": null
    }}
]

IMPORTANT: Return ONLY the JSON array. No additional text or explanation."""

    def _parse_llm_response(self, response_text: str) -> Optional[List[Dict[str, Any]]]:
        """Parse and validate LLM response with improved error handling."""
        try:
            # Remove any markdown code blocks if present
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON
            test_cases = json.loads(cleaned_text)
            
            # Validate structure
            if not isinstance(test_cases, list):
                raise ValueError("Response is not a JSON array")
            
            # Validate and normalize each test case
            validated_cases = []
            for idx, case in enumerate(test_cases, 1):
                required_fields = {"test_case", "description", "expected_result", "input"}
                if not all(field in case for field in required_fields):
                    logging.warning(f"Test case {idx} missing required fields, skipping")
                    continue
                
                # Normalize expected_result to Pass/Fail
                case["expected_result"] = "Pass" if case["expected_result"].lower() == "pass" else "Fail"
                validated_cases.append(case)
            
            return validated_cases

        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error parsing response: {str(e)}")
            return None

    def generate_test_cases(self, rules_file: str, output_file: str) -> None:
        """Main method to generate and save test cases."""
        try:
            # Load rules
            with open(rules_file, "r") as f:
                rules = json.load(f)

            all_test_cases = {}
            total_fields = sum(len(details["fields"]) for details in rules.values())
            processed_fields = 0

            # Process each field
            for parent_field, details in rules.items():
                for field_name, field_details in details["fields"].items():
                    full_field_name = f"{parent_field}.{field_name}"
                    logging.info(f"Processing field {processed_fields + 1}/{total_fields}: {full_field_name}")

                    # Generate prompt
                    prompt = self._generate_prompt(
                        field_name,
                        field_details["data_type"],
                        field_details["constraints"]
                    )

                    # Get LLM response with retries
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            response = self.llm_client.generate_content(prompt)
                            test_cases = self._parse_llm_response(response.text)
                            
                            if test_cases:
                                all_test_cases[full_field_name] = test_cases
                                logging.info(f"Successfully generated {len(test_cases)} test cases")
                                break
                            else:
                                logging.warning(f"Attempt {attempt + 1}: Failed to generate valid test cases")
                        except Exception as e:
                            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
                            if attempt == max_retries - 1:
                                logging.error(f"Failed to generate test cases for {full_field_name} after {max_retries} attempts")
                    
                    processed_fields += 1

            # Save results
            self._save_test_cases(all_test_cases, output_file)
            
            # Generate summary
            self._generate_summary(all_test_cases, output_file)

        except Exception as e:
            logging.error(f"Failed to generate test cases: {str(e)}")
            raise

    def _save_test_cases(self, test_cases: Dict[str, List[Dict[str, Any]]], output_file: str) -> None:
        """Save test cases with backup."""
        try:
            # Create backup of existing file if it exists
            if os.path.exists(output_file):
                backup_file = f"{output_file}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
                os.rename(output_file, backup_file)
                logging.info(f"Created backup: {backup_file}")

            # Save new test cases
            with open(output_file, "w") as f:
                json.dump(test_cases, f, indent=2)
            logging.info(f"Successfully saved test cases to {output_file}")

        except Exception as e:
            logging.error(f"Failed to save test cases: {str(e)}")
            raise

    def _generate_summary(self, test_cases: Dict[str, List[Dict[str, Any]]], output_file: str) -> None:
        """Generate a summary of the test case generation."""
        total_fields = len(test_cases)
        total_test_cases = sum(len(cases) for cases in test_cases.values())
        
        summary = (
            f"\nTest Case Generation Summary\n"
            f"{'='*30}\n"
            f"Total fields processed: {total_fields}\n"
            f"Total test cases generated: {total_test_cases}\n"
            f"Average test cases per field: {total_test_cases/total_fields:.2f}\n"
            f"Output file: {output_file}\n"
            f"{'='*30}"
        )
        
        logging.info(summary)

def main():
    try:
        generator = TestCaseGenerator()
        generator.generate_test_cases(
            generator.config["constrains_processed_rules_file"],
            generator.config["generated_test_cases_file"]
        )
    except Exception as e:
        logging.error(f"Application failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()