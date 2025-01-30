#standard
import os
import pandas as pd
import logging

#for getting the key
from dotenv import load_dotenv

# LangChain-related imports
from langchain_openai.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# THIS IS USED TO PREP THE CSV FILES FOR THE DIFFERENT MISTAKES

def load_csv(file_path):
    #load event logs into dataframe
    df = pd.read_csv(file_path, delimiter=',')
    logger.info(f"Loaded DataFrame with shape: {df.shape}")
    return df



def main():
    # File path to the event logs CSV
    file_path = "Pizza_Case_CSV.csv"

    # Step 1: Load the raw data
    df = load_csv(file_path)

    # Step 2: Create a shared context
    shared_context = {"df": df}

    # Step 3: Create the analysis agent with the shared context
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY"))
    agent = create_pandas_dataframe_agent(llm, shared_context["df"], verbose=True, allow_dangerous_code=True)

    # Step 4: Clean and validate the DataFrame using the agent
    question = """
    Clean and validate this DataFrame, that represents an event log. Follow these steps: 
    1. Manipulate the dataframe that the activities "Start preparing pizza" "Start baking pizza" "Baking pizza ready" are all converted to "Make Pizza"
    2. save the updated dataframe in Pizza_Case_CSV_I22
    """
    try:
        response = agent.invoke({"input": question})
        print("Agent Response:")
        print(response)

        # Retrieve the updated DataFrame from the shared context
        updated_df = shared_context["df"]
        logger.info(f"Updated DataFrame shape: {updated_df.shape}")

        # Save the updated DataFrame
        output_file_path = "./Pizza_Case_CSV_I22.csv"
        updated_df.to_csv(output_file_path, index=False)
        logger.info(f"Cleaned DataFrame saved to {output_file_path}")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
