
import pandas as pd
import ollama
from pathlib import Path

# ----------------------------------
# 1. Set project and dataset paths
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

INPUT_FILE = DATASET_DIR / "prompts.csv"
OUTPUT_FILE = DATASET_DIR / "llm_responses.csv"

# ----------------------------------
# 2. Check that the input file exists
# ----------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}\n"
        "Please create prompts.csv inside the dataset folder."
    )

# ----------------------------------
# 3. Read the input CSV
# ----------------------------------
prompts = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

# Normalize column names (Domain -> domain)
prompts.columns = prompts.columns.str.strip().str.lower()

# Check required columns
required_columns = {"prompt_id", "domain", "prompt"}
missing_columns = required_columns - set(prompts.columns)

if missing_columns:
    raise ValueError(
        f"Missing CSV columns: {missing_columns}. "
        f"Found columns: {prompts.columns.tolist()}"
    )

if prompts.empty:
    raise ValueError("The prompts.csv file contains no prompts.")

# ----------------------------------
# 4. Configure the Ollama model
# ----------------------------------
model = "llama3.2"

# ----------------------------------
# 5. Generate responses
# ----------------------------------
responses = []

for index, row in prompts.iterrows():
    prompt = str(row["prompt"]).strip()

    if not prompt:
        print(f"Skipping empty prompt at row {index + 1}.")
        continue

    print(f"Processing prompt {index + 1} of {len(prompts)}...")

    try:
        # Send the prompt to the local Ollama model
        ollama_response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Extract the generated answer
        answer = ollama_response["message"]["content"]

        # Store the prompt and its response
        responses.append({
            "prompt_id": row["prompt_id"],
            "domain": row["domain"],
            "prompt": prompt,
            "response": answer,
            "status": "success"
        })

        print("Response generated successfully.")

    except Exception as error:
        print(f"Error processing prompt {index + 1}: {error}")

        # Record the failure so the other prompts can continue
        responses.append({
            "prompt_id": row["prompt_id"],
            "domain": row["domain"],
            "prompt": prompt,
            "response": "",
            "status": f"error: {error}"
        })

# ----------------------------------
# 6. Save all responses to a CSV
# ----------------------------------
result = pd.DataFrame(
    responses,
    columns=[
        "prompt_id",
        "domain",
        "prompt",
        "response",
        "status"
    ]
)

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ----------------------------------
# 7. Display completion message
# ----------------------------------
successful = (result["status"] == "success").sum()

print("\n----------------------------------")
print("Ollama Local Interface - Completed")
print("----------------------------------")
print(f"Total prompts: {len(prompts)}")
print(f"Successful responses: {successful}")
print(f"Output file: {OUTPUT_FILE}")
print("Responses saved successfully.")