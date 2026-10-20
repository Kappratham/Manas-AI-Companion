"""
Manas Deployment Pipeline
- Loads base_short model
- Runs crisis classifier pre-processing
- Generates response with corrected evaluation
"""

import asyncio
from crisis_classifier import detect_crisis, insert_helpline
from eval.conversation_runner import UserSimulator
from auto_eval import load_manas_model

async def run_manas(user_input, max_turns=10):
    """
    Main deployment function for Manas.

    Args:
        user_input: First user message
        max_turns: Maximum conversation turns

    Returns:
        List of assistant responses
    """
    # Load model
    manas = load_manas_model(max_new_tokens=256, temperature=0.6)
    user_sim = UserSimulator(api_key=None)

    # System prompt with crisis detection
    system_prompt = open('adapted_data/manas_system_prompt_short_crisis_v2.txt',
                        'r', encoding='utf-8').read()

    messages = [{'role': 'user', 'content': user_input}]
    responses = []

    for turn in range(max_turns):
        # Run model
        result = await run_single_conversation(
            manas=manas,
            user_sim=user_sim,
            category="general",
            persona_id="p01",
            situation=user_input if turn == 0 else None,
            num_turns=1,
            scenario_id=f"deploy_{turn}",
            system_prompt=system_prompt,
            messages=messages,
            dry_run=False,
            skip_user_sim=(turn > 0)
        )

        # Get assistant response
        assistant_msgs = [m for m in result.messages if m['role'] == 'assistant']
        if assistant_msgs:
            response = assistant_msgs[0]['content']

            # Check for crisis and insert helpline if needed
            is_crisis, matches = detect_crisis(user_input)
            if is_crisis:
                response = insert_helpline(response, is_crisis)

            responses.append(response)
            messages.append({'role': 'assistant', 'content': response})

        # Get next user input
        if turn < max_turns - 1:
            next_input = await user_sim.generate_next_message(messages)
            messages.append({'role': 'user', 'content': next_input})
            user_input = next_input

    return responses

if __name__ == '__main__':
    # Test deployment
    test_input = "I don't see the point anymore. Nothing matters."
    print(f"User: {test_input}")

    # Run crisis detection
    is_crisis, matches = detect_crisis(test_input)
    print(f"Crisis detected: {is_crisis} (matches: {matches})")

    print("\nDeployment pipeline ready.")
    print("Usage: from deployment_pipeline import run_manas")
