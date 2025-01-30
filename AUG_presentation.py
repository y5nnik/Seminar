# standard
import os
import pandas as pd
import logging

# For getting the key
from dotenv import load_dotenv

# LangChain-related imports
from langchain_openai.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_csv(file_path):
    """Load event logs into a DataFrame."""
    df = pd.read_csv(file_path, delimiter=';')
    logger.info(f"Loaded DataFrame with shape: {df.shape}")
    return df

def main():
    # 1. Load the raw data
    file_path = "./Pizza_Case_Real_small_pres.csv"
    df = load_csv(file_path)

    # 2. Create a shared context (dictionary)
    shared_context = {"df": df}

    # 3. Create the analysis agent with the shared DataFrame
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    agent = create_pandas_dataframe_agent(
        llm,
        shared_context["df"],
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        max_chain_length=10,
    )

    # 4. Clean and validate the DataFrame using the agent
    question = """
    Clean and validate this DataFrame. Follow these steps:
    1. Check for missing values in critical columns (e.g., 'case_id', 'activity', 'timestamp') 
       and fill them appropriately. For this, look at the line above and below and intuitively 
       decide what to put in the missing space.
    2. Drop columns that are entirely empty or irrelevant for analysis.
    3. Inspect the unique values in each column to check for spelling errors or typos 
       (e.g., inconsistent capitalization, extra spaces, or unlikely outliers). 
       Fix any typos you find.
    4. Remove duplicate rows, if any exist.
    5. Validate that there are no empty rows.
    6. Save the updated dataframe, as ./Pizza_Case_Real_cleaned_pres.csv
    """

    try:
        # Use agent.run(...) instead of agent.invoke(...) 
        # so the DataFrame remains defined across all steps.
        response = agent.run(question)

        # The agent modifies its internal copy of the DataFrame; retrieve the updated copy:
        updated_df = agent.df

        logger.info(f"Agent response:\n{response}")
        logger.info(f"Updated DataFrame shape: {updated_df.shape}")

        # Save the updated DataFrame
        output_file_path = "./Pizza_Case_Real_small_cleaned_pres.csv"
        updated_df.to_csv(output_file_path, index=False)
        logger.info(f"Cleaned DataFrame saved to {output_file_path}")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
