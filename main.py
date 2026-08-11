# -*- coding: utf-8 -*-
import os
import json
import argparse
import pandas as pd
import requests
import concurrent.futures
from tqdm import tqdm


# Fetch the port dynamically passed from the SLURM script setup
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8080)
args = parser.parse_args()

MODEL_ID = "unsloth/gemma-4-31B-it-GGUF" 
GEN_MAX_NEW_TOKENS = 8192
SAVE_EVERY = 50
API_URL = f"http://localhost:{args.port}/v1/chat/completions"

PROMPT_TEMPLATE = """[PROMPT TEMPLATE]"""

def main():
    print(f"Connecting to inference server at: {API_URL}", flush=True)

    # ----------------------- PREPROCESSING -----------------------
    def _ensure_column(df: pd.DataFrame, col: str):
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df))

    def process_csv_with_llm(
        csv_file_name: str,
        prompt_column: str,
        output_column: str,
        prompt_template: str,
        total_lines_to_process: int,
        run: str,
        prompt_type: str,
        id_column: str = None,
        rerun_ids: list = None,
        save_every: int = SAVE_EVERY,
    ):
        try:
            df = pd.read_csv("./data/" + csv_file_name)
            _ensure_column(df, output_column)
            normalized_model_id = MODEL_ID.replace("/", "-")
            output_filename = f"{csv_file_name.replace(".csv", "")}-{prompt_type}-{normalized_model_id}-run{run}.csv"

            unprocessed_mask = df[output_column].isna() | (df[output_column] == "")

            if rerun_ids is not None and id_column is not None:
                if id_column not in df.columns:
                    raise KeyError(f"Specified ID column '{id_column}' not found in the dataset.")

                id_mask = df[id_column].astype(str).isin([str(x) for x in rerun_ids])

                final_mask = unprocessed_mask & id_mask
                print(f"Rerun mode activated. Matching IDs found: {id_mask.sum()}", flush=True)
            else:
                final_mask = unprocessed_mask

            indices_to_process = df.index[final_mask].tolist()
            print(f"Remaining rows to process: {len(indices_to_process)}", flush=True)

            payload_base = {
                "model": MODEL_ID,
                "temperature": 0.0,
                "max_tokens": GEN_MAX_NEW_TOKENS,
            }
            
            # 1. Define a worker function for a single row execution
            def process_row(idx):
                content = df.loc[idx, prompt_column]
                messages = [{"role": "user", "content": prompt_template.format(content=content)}]
                payload = {**payload_base, "messages": messages}
                
                try:
                    response = requests.post(
                        API_URL, 
                        json=payload, 
                        headers={"Content-Type": "application/json"},
                        timeout=300
                    )
                    response_data = response.json()
                    generation = response_data["choices"][0]["message"]["content"]
                    return idx, generation
                except Exception as api_err:
                    print(f"\nFailed row index {idx} due to API Error: {api_err}", flush=True)
                    return idx, None
                    
            # 2. Use ThreadPoolExecutor to hit the 4 parallel slots
            # Setting max_workers=4 matches llama.cpp n_parallel configuration
            MAX_WORKERS = 4 
            
            print(f"Starting parallel processing with {MAX_WORKERS} workers...", flush=True)
            
            completed_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Map the worker function to all indices
                futures = {executor.submit(process_row, idx): idx for idx in indices_to_process}
                
                # As each request finishes, update the dataframe and save checkpoints
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    idx, generation = future.result()
                    
                    if generation is not None:
                        df.at[idx, output_column] = generation
                    
                    completed_count += 1
                    # Periodic saving based on completed counts
                    if completed_count % save_every == 0:
                        df.to_csv("./output/" + output_filename, index=False)
                        print(f"Saved intermediate checkpoints. Completed: {completed_count}", flush=True)

            df.to_csv("./output/" + output_filename, index=False)
            print(f"? Finished processing run {run} for template {prompt_type}.")
        except FileNotFoundError:
            print(f"Error: file '{csv_file_name}' not found.")
        except KeyError as ke:
            print(f"Error: column missing in CSV -> {ke}")
        except Exception as e:
            print(f"Error: {e}")
            raise

    experiments = [
        {"run": "1", "prompt_template": PROMPT_TEMPLATE, "prompt_type": "[prompt-type]"},
        {"run": "2", "prompt_template": PROMPT_TEMPLATE, "prompt_type": "[prompt-type]"},
    ]

    for experiment in experiments:
        process_csv_with_llm(
            csv_file_name="[input-data].csv",
            prompt_column="[input-column]",
            output_column="[output-column]",
            prompt_template=experiment.get("prompt_template"),
            total_lines_to_process=[number-of-rows],
            run=experiment.get("run"),
            prompt_type=experiment.get("prompt_type"),
            # id_column="[id-column]",
            # rerun_ids=[],
        )

if __name__ == '__main__':
    main()
