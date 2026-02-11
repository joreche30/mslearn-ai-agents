import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------
# Add references (per file2 instructions)
# ---------------------------------------------
from agent_framework import AgentThread, ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field
from typing import Annotated


async def process_expenses_data(prompt: str, expenses_data: str) -> None:
    """
    Create an agent and process the user's prompt against the provided expenses data.
    The Foundry project endpoint and model deployment name are taken from .env
    (PROJECT_ENDPOINT and AZURE_AI_MODEL_DEPLOYMENT_NAME).
    """

    # ---------------------------------------------
    # Create a chat agent (per file2 instructions)
    # ---------------------------------------------
    async with (
        AzureCliCredential() as credential,
        ChatAgent(
            chat_client=AzureAIAgentClient(credential=credential),
            name="expenses_agent",
            instructions="""You are an AI assistant for expense claim submission. When a user submits expenses data and requests an expense claim, use the plug-in function to send an email to expenses@contoso.com with the subject 'Expense Claim' and a body that contains itemized expenses with a total. Then confirm to the user that you've done so.""",
            tools=send_email,
        ) as agent,
    ):
        # ---------------------------------------------
        # Use the agent to process the expenses data
        # (per file2 instructions)
        # ---------------------------------------------
        try:
            # Add the input prompt to a list of messages to be submitted
            prompt_messages = [f"{prompt}: {expenses_data}"]

            # Invoke the agent with the messages
            response = await agent.run(prompt_messages)

            # Display the response
            print(f"\n# Agent:\n{response}")

        except Exception as e:
            # Something went wrong
            print(e)


# ---------------------------------------------
# Create a tool function for the email functionality
# (per file2 instructions)
# ---------------------------------------------
def send_email(
    to: Annotated[str, Field(description="Who to send the email to")],
    subject: Annotated[str, Field(description="The subject of the email.")],
    body: Annotated[str, Field(description="The text body of the email.")],
):
    # Note: This simulates sending an email by printing it to the console.
    # In a real application, you'd integrate with an email service.
    print("\nTo:", to)
    print("Subject:", subject)
    print(body, "\n")


async def main() -> None:
    # Clear console and load environment
    os.system("cls" if os.name == "nt" else "clear")
    load_dotenv()

    # Locate and read the sample expenses data that ships with the lab
    script_dir = Path(__file__).parent
    # The lab says “a file containing expenses data” exists; repositories often
    # use CSV for this step. If your folder uses a different filename, update below.
    candidate_files = ["expenses.csv", "data.txt", "expenses.txt"]
    data_file = next((f for f in candidate_files if (script_dir / f).exists()), None)
    if not data_file:
        raise FileNotFoundError(
            f"Could not find an expenses data file. Checked: {candidate_files}"
        )

    with (script_dir / data_file).open("r", encoding="utf-8") as f:
        data = f.read().strip()

    print("Here is the expenses data in your file:\n")
    print(data, "\n")

    # Simple loop to accept a single task (matches the lab’s flow)
    user_prompt = input("What would you like me to do with it?\n")
    if user_prompt.strip():
        await process_expenses_data(user_prompt.strip(), data)


if __name__ == "__main__":
    asyncio.run(main())
