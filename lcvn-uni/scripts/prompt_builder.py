from typing import Dict, Tuple


def build_instruction_prompt(
    start_pose_str: str, 
    dxy_range: Tuple[float, float], 
    dyaw_range: Tuple[float, float], 
    instruction_str: str
) -> str:
    return (
        "Task: Integrated Navigation Prediction (Action and Observation from Instruction)\n"
        "Description: Based on the current first-person observation, starting point observation and coordinate, and a given navigation instruction, perform an integrated prediction of the next navigation step. This involves reasoning about the optimal action to follow the instruction and simultaneously visualizing the resulting first-person view as a single, seamless process where the action directly causes the visual change.\n\n"
        "1. If the instruction is completed or the goal is reached, output the command 'Stop'. Otherwise, define the move using three components:\n"
        "    - dx: displacement along the agent's facing direction,\n"
        "    - dy: displacement perpendicular to the facing direction,\n"
        "    - dyaw: change in heading angle (i.e., how much the agent rotates).\n"
        "2. All components are discretized into bin tokens: for example,\n"
        "    - `dx pos bin 02`: dx = +0.02 meters,\n"
        "    - `dy neg bin 23`: dy = -0.23 meters,\n"
        "    - `dyaw pos bin 26`: counterclockwise rotation of +0.26 radians.\n"
        "3. Spatial Interpretation: Understanding these values is key to predicting the next image.\n"
        "    - The magnitude of [dx, dy] reflects how far the agent moves in this step — larger values indicate greater positional shift, leading to larger visual changes.\n"
        "    - dyaw controls the agent's rotation (change in heading). A positive dyaw indicates a left turn (counter-clockwise), while a negative dyaw indicates a right turn (clockwise).\n"
        "4. Value Ranges: The numeric values for the move components fall within these ranges:\n"
        f"    - Range of dx, dy: [{dxy_range[0]:.2f}, {dxy_range[1]:.2f}]\n"
        f"    - Range of dyaw: [{dyaw_range[0]:.2f}, {dyaw_range[1]:.2f}]\n\n"
        "Inputs:\n"
        "- Start Observation: <image>\n"
        "- Current Observation: <image>\n"
        f"{start_pose_str}\n"
        f"Instruction: {instruction_str}.\n\n"
        "Required Output Format:\n"
        "Your output must contain the action text first, and then be IMMEDIATELY followed by the predicted image itself.\n"
        "Example of the TEXT PART: 'Move by dx: <dx_bin_10>, dy: <dy_bin_00>, dyaw: <dyaw_bin_05>'"
    )