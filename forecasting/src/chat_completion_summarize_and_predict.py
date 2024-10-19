import os
import json
from typing import List, Optional
from llama import Llama
import fire
from tqdm import tqdm
from pathlib import Path
def extract_key_data(player_data):
    """Extract key scalar data from the player's data."""
    return {
        "Score": player_data["scalars"].get("score_float"),
        "Army size": player_data["scalars"].get("popArmy"),
        "Total unit value": player_data["scalars"].get("total_value_units"),
        "Collected minerals": player_data["scalars"].get("collected_minerals"),
        "Collected vespene": player_data["scalars"].get("collected_vespene"),
        "Spent minerals": player_data["scalars"].get("spent_minerals"),
        "Spent vespene": player_data["scalars"].get("spent_vespene"),
        "Damage dealt life": player_data["scalars"].get("total_damage_dealt_life"),
        "Idle production time": player_data["scalars"].get("idle_production_time"),
        "Idle worker time": player_data["scalars"].get("idle_worker_time")
    }

def create_initial_prompt():
    """Create the initial prompt to set the context for the model."""
    prompt = """
You are an AI tasked with summarizing StarCraft II games based on time-series data for two players. 
Your goal is to analyze the game progression by comparing the economic strength, army size, unit value, 
and strategic choices made by both players. For each time step, you will be provided data for Player 0 and Player 1. 

Please do the following:
- Compare their economy, army size, and resource usage.
- Identify key strategic differences between the players.
- Predict which player might be in a stronger position at each time step.
You will receive one time step's data at a time, followed by a summary prompt.
Begin the analysis once you receive the data.

At each step, you will receive a game state for both Player 0 and Player 1. Summarize the state at that time and provide an analysis.
    """
    return prompt

def create_prompt_for_time_step(player0_data, player1_data, time_step):
    """Create a summary prompt for both players at a given time step."""
    player0_summary = extract_key_data(player0_data)
    player1_summary = extract_key_data(player1_data)

    prompt = f"Here is the data for both players at {time_step} minutes:\n\n"

    prompt += f"**Player 0:**\n"
    for key, value in player0_summary.items():
        prompt += f"- {key}: {value}\n"

    prompt += f"\n**Player 1:**\n"
    for key, value in player1_summary.items():
        prompt += f"- {key}: {value}\n"

    prompt += "\nPlease provide a summary of the game state and the strategic situation based on this data. Highlight the key differences between the players, suggest possible next steps for each, and predict who might have the advantage at this point in time.\n"

    return prompt

def create_final_prediction_prompt(summaries):
    """Create a final prediction prompt based on the generated summaries."""
    prompt = "Based on the following game summaries, predict which player will win the game:\n\n"
    
    for idx, summary in enumerate(summaries):
        prompt += f"Time Step {idx + 1} Summary:\n{summary}\n\n"
    
    prompt += "Now, predict the winner of the game (Player 0 or Player 1) and provide reasoning for your prediction. The first character of the output should be 0 or 1 for which player will win."
    
    return prompt

def generate_summaries(game_data, llama_model, max_gen_len, temperature, top_p):
    """Generate game summaries using the LLaMA model at each time step."""
    results = []
    
    # Step 1: Send the initial context-setting prompt
    initial_prompt = create_initial_prompt()
    initial_input = [{"role": "user", "content": initial_prompt}]
    llama_model.chat_completion([initial_input], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
    game_data_ = game_data["game"]
    # Iterate through game steps and pair Player 0 and Player 1 data
    for idx in range(0, len(game_data_), 2):
    # for idx in tqdm(range(0, len(game_data_), 2), total=len(game_data_) // 2):
        player0_data = game_data_[idx]
        player1_data = game_data_[idx + 1]

        time_step = player0_data["game_step"].split(":")[0]  # Assuming both players share the same time_step
        # print(time_step)
        if float(time_step) >= 15:
            # Create a prompt for the current time step
            prompt = create_prompt_for_time_step(player0_data, player1_data, time_step)

            # Generate summary from the LLaMA model for the current time step
            dialog_input = [{"role": "user", "content": prompt}]
            responses = llama_model.chat_completion(
                [dialog_input],  # Input as a single dialog
                max_gen_len=max_gen_len, 
                temperature=temperature, 
                top_p=top_p
            )

            # Append the generated summary to results
            result = f"Time Step {time_step} Summary:\n{responses[0]['generation']['content']}\n"
            results.append(result)
            break

    # Step 2: Generate a final prediction based on the cumulative summaries
    final_prediction_prompt = create_final_prediction_prompt(results)
    dialog_input = [{"role": "user", "content": final_prediction_prompt}]
    
    # Generate final prediction
    prediction_response = llama_model.chat_completion(
        [dialog_input], 
        max_gen_len=max_gen_len, 
        temperature=temperature, 
        top_p=top_p
    )
    
    final_prediction = prediction_response[0]['generation']['content']
    results.append(f"Final Prediction: {final_prediction}\n")
    print(final_prediction)
    try:
        idx0 = final_prediction.index("0")
    except ValueError:
        idx0 = 100000000000000000000000000000000000000000000000
    try:
        idx1 = final_prediction.index("1")
    except ValueError:
        idx1 = 1000000000000000000000000000000000000000000000000

    predicted = "1" if idx1 < idx0 else "0"
    # assert predicted in ["0", "1"], f"Invalid prediction: {predicted}"
    correct_answer = str(game_data["win"][0].index(0.0))
    is_correct = predicted == correct_answer

    return results, is_correct
    

def main(
    game_data_file: str,
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 5120,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
    temperature: float = 0.6,
    top_p: float = 0.9
):
    """Main function to run game data summarization using the LLaMA model."""
    # Load the LLaMA model
    llama_model = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    correct, total = 0,0

    for i in tqdm(Path("/mnt/fast/").glob("*.json"), total=100):

        print(i)
        # break

        # Load the game data JSON file
        with open(i, "r") as f:
            game_data = json.load(f)

        # Generate the summaries
        results, c = generate_summaries(
            game_data=game_data, 
            llama_model=llama_model, 
            max_gen_len=max_gen_len, 
            temperature=temperature, 
            top_p=top_p
        )
        # print("correct", c)
        correct += c
        total += 1

        # Output the results to the terminal
        # for result in results:
        #     print(result)
    
        print(correct, total)

if __name__ == "__main__":
    fire.Fire(main)
