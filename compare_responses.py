
import pandas as pd
from pathlib import Path

# Find the project and dataset folders
BASE_DIR = Path(__file__).resolve().parent

if (BASE_DIR / "Dataset").is_dir():
    DATASET_DIR = BASE_DIR / "Dataset"
else:
    DATASET_DIR = BASE_DIR / "dataset"

# Read the existing LLM responses
input_file = DATASET_DIR / "llm_responses.csv"
output_file = DATASET_DIR / "response_comparison.csv"

responses = pd.read_csv(input_file)

# Check the required columns
required_columns = ["prompt", "response"]
missing = [col for col in required_columns
           if col not in responses.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        f"Available columns: {list(responses.columns)}"
    )

# Prepare one comparison row for each response
comparison = pd.DataFrame({
    "Prompt": responses["prompt"],
    "Response": responses["response"],
    "Quality": "",
    "Accuracy": "",
    "Relevance": "",
    "Observations": ""
})

# Save the comparison file
comparison.to_csv(output_file, index=False)

print(f"Comparison file saved: {output_file}")
print(f"Total responses prepared: {len(comparison)}")
print(comparison[["Prompt", "Quality", "Accuracy", "Relevance"]])