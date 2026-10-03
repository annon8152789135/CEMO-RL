import pandas as pd
import ast


# ============================================================
# Configuration
# ============================================================

input_file = "action_differences_0_5.csv"
output_file = "action_differences_0_5.txt"

# Change this if the action encoding is different
ACTION_NAMES = {
    0: "left",
    1: "right"
}


# ============================================================
# Helper functions
# ============================================================

def parse_transitions(value):
    """
    Convert the string stored in only_g1 / only_g2
    into a Python list of dictionaries.
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def format_state(state):
    """
    Make the state description slightly more natural.

    Examples:
        'lvl equal to 3'
            -> "level is equal to 3"

        'pos between 31 and 15'
            -> "position is between 31 and 15"
    """

    state = str(state).strip()

    # Make feature names easier to read
    state = state.replace("lvl", "level")
    state = state.replace("pos", "position")

    # Add "is" for common predicate forms
    if " between " in state and " is between " not in state:
        state = state.replace(" between ", " is between ", 1)

    if " equal to " in state and " is equal to " not in state:
        state = state.replace(" equal to ", " is equal to ", 1)

    if " greater than " in state and " is greater than " not in state:
        state = state.replace(" greater than ", " is greater than ", 1)

    if " less than " in state and " is less than " not in state:
        state = state.replace(" less than ", " is less than ", 1)

    return state


def format_action(action):
    """
    Convert action ID into readable text.
    """
    return ACTION_NAMES.get(action, f"action {action}")


def format_probability(probability):
    """
    Format probability without unnecessary trailing zeros.

    0.900000 -> 0.9
    0.056603 -> 0.057
    """
    return f"{float(probability):.3f}".rstrip("0").rstrip(".")


# ============================================================
# Read CSV
# ============================================================

df = pd.read_csv(input_file)

sentences = []


# ============================================================
# Generate explanations
# ============================================================

for _, row in df.iterrows():

    state = format_state(row["state"])

    transitions_a = parse_transitions(row["only_g1"])
    transitions_b = parse_transitions(row["only_g2"])

    # --------------------------------------------------------
    # Both agents have a differing transition
    # --------------------------------------------------------

    if transitions_a and transitions_b:

        # If there is more than one transition, use the one
        # with the highest probability for each agent.
        transition_a = max(
            transitions_a,
            key=lambda x: x.get("probability", 0)
        )

        transition_b = max(
            transitions_b,
            key=lambda x: x.get("probability", 0)
        )

        probability_a = format_probability(
            transition_a["probability"]
        )

        probability_b = format_probability(
            transition_b["probability"]
        )

        action_a = format_action(
            transition_a["action"]
        )

        action_b = format_action(
            transition_b["action"]
        )

        sentence = (
            f"When the agent's {state}, "
            f"Agent A has a {probability_a} probability of going {action_a}, "
            f"while Agent B has a {probability_b} probability of going {action_b}."
        )

    # --------------------------------------------------------
    # Only Agent A has a differing transition
    # --------------------------------------------------------

    elif transitions_a:

        transition_a = max(
            transitions_a,
            key=lambda x: x.get("probability", 0)
        )

        probability_a = format_probability(
            transition_a["probability"]
        )

        action_a = format_action(
            transition_a["action"]
        )

        sentence = (
            f"When the agent's {state}, "
            f"Agent A has a {probability_a} probability of going {action_a}, "
            f"while no corresponding differing transition is recorded for Agent B."
        )

    # --------------------------------------------------------
    # Only Agent B has a differing transition
    # --------------------------------------------------------

    elif transitions_b:

        transition_b = max(
            transitions_b,
            key=lambda x: x.get("probability", 0)
        )

        probability_b = format_probability(
            transition_b["probability"]
        )

        action_b = format_action(
            transition_b["action"]
        )

        sentence = (
            f"When the agent's {state}, "
            f"no corresponding differing transition is recorded for Agent A, "
            f"while Agent B has a {probability_b} probability of going {action_b}."
        )

    # --------------------------------------------------------
    # Neither side has a transition
    # --------------------------------------------------------

    else:
        continue

    sentences.append(sentence)


# ============================================================
# Save TXT
# ============================================================

with open(output_file, "w", encoding="utf-8") as file:
    for sentence in sentences:
        file.write(sentence + "\n")

print(f"Saved {len(sentences)} explanations to: {output_file}")