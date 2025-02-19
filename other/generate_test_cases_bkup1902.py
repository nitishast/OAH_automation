import json
import os
from typing import Dict, List, Optional, Any
import google.generativeai as genai
import yaml
from datetime import datetime
import logging
import re

class TestCaseGenerator:
    def __init__(self, config_path: str = "config.yaml"):
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
        constraint_list = "\n".join([f"- {c}" for c in constraints])
        return f"""
Generate a JSON array of test cases for field '{field_name}'. 
Specifications:
- Data Type: {data_type}
- Constraints:
{constraint_list}

Important Rules:
1. Return ONLY valid JSON array
2. NO explanatory text before or after the JSON
3. NO code block markers (```)
4. Each test case must have these fields:
   - "test_case": String identifier (e.g., "TC001_Valid_Basic")
   - "description": String describing the test
   - "expected_result": EXACTLY "Pass" or "Fail"
   - "input": The test value (can be null, string, number, etc.)

Include these test categories:
1. Valid inputs (basic, complex)
2. Invalid inputs (null, empty)
3. Type validation
4. Boundary conditions
5. Edge cases

Example format:
[
    {{
        "test_case": "TC001_Valid_Basic",
        "description": "Basic valid input test",
        "expected_result": "Pass",
        "input": "example"
    }}
]"""

    def _clean_json_response(self, response_text: str) -> str:
        """Clean the LLM response text to ensure valid JSON."""
        # Remove any markdown code blocks
        cleaned = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', response_text, flags=re.DOTALL)
        
        # Remove any non-JSON text before or after the array
        cleaned = re.sub(r'^[^[]*(\[.*\])[^]]*$', r'\1', cleaned, flags=re.DOTALL)
        
        # Fix common escape character issues
        cleaned = cleaned.replace('\\n', '\\\\n')
        cleaned = cleaned.replace('\\t', '\\\\t')
        
        # Fix unescaped quotes
        cleaned = re.sub(r'(?<!\\)"(?=.*".*})', '\\"', cleaned)
        
        # Remove any invalid control characters
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
        
        return cleaned.strip()

    def _parse_llm_response(self, response_text: str) -> Optional[List[Dict[str, Any]]]:
        """Parse and validate LLM response with improved error handling."""
        try:
            # Clean the response
            cleaned_text = self._clean_json_response(response_text)
            
            # Log the cleaned text for debugging
            logging.debug(f"Cleaned JSON:\n{cleaned_text}")
            
            # Parse JSON
            test_cases = json.loads(cleaned_text)
            
            if not isinstance(test_cases, list):
                raise ValueError("Response is not a JSON array")
            
            # Validate and normalize test cases
            validated_cases = []
            for idx, case in enumerate(test_cases, 1):
                # Validate required fields
                required_fields = {"test_case", "description", "expected_result", "input"}
                if not all(field in case for field in required_fields):
                    logging.warning(f"Test case {idx} missing required fields, skipping")
                    continue
                
                # Normalize expected_result
                case["expected_result"] = "Pass" if case["expected_result"].lower() == "pass" else "Fail"
                
                # Ensure test_case is properly formatted
                if not case["test_case"].startswith("TC"):
                    case["test_case"] = f"TC{str(idx).zfill(3)}_{case['test_case']}"
                
                validated_cases.append(case)
            
            return validated_cases

        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {str(e)}")
            logging.debug(f"Failed JSON:\n{response_text}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error parsing response: {str(e)}")
            return None

    def generate_test_cases(self, rules_file: str, output_file: str) -> None:
        """Main method to generate and save test cases with improved retry logic."""
        try:
            # Load rules
            with open(rules_file, "r") as f:
                rules = json.load(f)

            all_test_cases = {}
            total_fields = sum(len(details["fields"]) for details in rules.values())
            processed_fields = 0

            for parent_field, details in rules.items():
                for field_name, field_details in details["fields"].items():
                    full_field_name = f"{parent_field}.{field_name}"
                    logging.info(f"Processing field {processed_fields + 1}/{total_fields}: {full_field_name}")

                    max_retries = 5  # Increased from 3 to 5
                    for attempt in range(max_retries):
                        try:
                            # Generate prompt with more specific constraints
                            prompt = self._generate_prompt(
                                field_name,
                                field_details["data_type"],
                                field_details["constraints"]
                            )

                            # Get LLM response
                            response = self.llm_client.generate_content(
                                prompt,
                                generation_config=genai.types.GenerationConfig(
                                    temperature=0.7,  # Add some randomness for retries
                                    max_output_tokens=2000
                                )
                            )

                            test_cases = self._parse_llm_response(response.text)
                            
                            if test_cases and len(test_cases) >= 5:  # Ensure minimum number of test cases
                                all_test_cases[full_field_name] = test_cases
                                logging.info(f"Successfully generated {len(test_cases)} test cases")
                                break
                            else:
                                logging.warning(f"Attempt {attempt + 1}: Generated insufficient test cases")
                                
                        except Exception as e:
                            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
                            if attempt == max_retries - 1:
                                logging.error(f"Failed to generate test cases for {full_field_name} after {max_retries} attempts")
                    
                    processed_fields += 1

            # Save results with backup
            self._save_test_cases(all_test_cases, output_file)
            
            # Generate summary
            self._generate_summary(all_test_cases, output_file)

        except Exception as e:
            logging.error(f"Failed to generate test cases: {str(e)}")
            raise

    def _save_test_cases(self, test_cases: Dict[str, List[Dict[str, Any]]], output_file: str) -> None:
        """Save test cases with backup and validation."""
        try:
            # Validate before saving
            if not test_cases:
                raise ValueError("No test cases to save")

            # Create backup
            if os.path.exists(output_file):
                backup_file = f"{output_file}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
                os.rename(output_file, backup_file)
                logging.info(f"Created backup: {backup_file}")

            # Save new test cases
            with open(output_file, "w") as f:
                json.dump(test_cases, f, indent=2, ensure_ascii=False)
            logging.info(f"Successfully saved test cases to {output_file}")

        except Exception as e:
            logging.error(f"Failed to save test cases: {str(e)}")
            raise

    def _generate_summary(self, test_cases: Dict[str, List[Dict[str, Any]]], output_file: str) -> None:
        """Generate a detailed summary of the test case generation."""
        total_fields = len(test_cases)
        total_test_cases = sum(len(cases) for cases in test_cases.values())
        
        summary = (
            f"\nTest Case Generation Summary\n"
            f"{'='*30}\n"
            f"Total fields processed: {total_fields}\n"
            f"Total test cases generated: {total_test_cases}\n"
            f"Average test cases per field: {total_test_cases/total_fields:.2f}\n"
            f"Output file: {output_file}\n\n"
            f"Field-wise breakdown:\n"
        )
        
        for field, cases in test_cases.items():
            summary += f"- {field}: {len(cases)} test cases\n"
        
        logging.info(summary)

if __name__ == "__main__":
    # Set up logging with more detail
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_generation.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        generator = TestCaseGenerator()
        generator.generate_test_cases(
            generator.config["constrains_processed_rules_file"],
            generator.config["generated_test_cases_file"]
        )
    except Exception as e:
        logging.error(f"Application failed: {str(e)}")
        raise