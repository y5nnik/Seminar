# standard
import os
import pandas as pd
import logging

# For getting the key
from dotenv import load_dotenv

# LangChain-related imports
from langchain_openai.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Mistake: there is a change in ownership after 01.06.2018 , so all cases before that are not relevant

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_csv(file_path):
    """Load event logs into a DataFrame."""
    df = pd.read_csv(file_path, delimiter=',')
    logger.info(f"Loaded DataFrame with shape: {df.shape}")
    return df

def main():
    # 1. Load the raw data
    file_path = "./Pizza_Case_CSV_I26.csv"
    df = load_csv(file_path)

    # 2. Create a shared context (dictionary)
    shared_context = {"df": df}

    # 3. Create the analysis agent with the shared DataFrame
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    agent = create_pandas_dataframe_agent(
        llm,
        shared_context["df"],
        verbose=True,
        allow_dangerous_code=True,
    )

    # 4. Clean and validate the DataFrame using the agent
    question = """
    Can you spot the quality issue "Irrelevant Cases" in the given dataframe df?
    This quality issue corresponds to the scenario where certain cases in an event log are deemed to be irrelevant for a particular context of analysis.
    In our case there is a change in ownership after 01.06.2018 , so all cases before that are not relevant and can be dropped.
    It is possible that the mistake type does not exist or can't be found without more infromation.
    If possible Fix the Event Log Quality Issue.
    If the dataframe has been changed, Save the updated dataframe, as ./Pizza_Case_CSV_informed_I26.csv
    """

    try:
        response = agent.invoke(question)

        # The agent modifies its internal copy of the DataFrame; retrieve the updated copy:
        updated_df = agent.df

        logger.info(f"Agent response:\n{response}")
        logger.info(f"Updated DataFrame shape: {updated_df.shape}")

        # Save the updated DataFrame
        output_file_path = "./Pizza_Case_CSV_informed_I26.csv"
        updated_df.to_csv(output_file_path, index=False)
        logger.info(f"Cleaned DataFrame saved to {output_file_path}")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
