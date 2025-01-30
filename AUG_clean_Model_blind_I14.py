# standard
import os
import pandas as pd
import logging

# for getting the key
from dotenv import load_dotenv

# LangChain-related imports
from langchain_openai.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


# Mistake: delete all time stamps, see if he notices or maybe just skip


def load_csv(file_path):
    """
    Load event logs into a DataFrame.
    """
    df = pd.read_csv(file_path, delimiter=';')
    logger.info(f"Loaded DataFrame with shape: {df.shape}")
    return df


def main():
    # File path to the event logs CSV
    file_path = "./Pizza_Case_Real_small_pres.csv"

    # Step 1: Load the raw data
    df = load_csv(file_path)

    # Step 2: Create a shared context
    shared_context = {"df": df}

    # Step 3: Create the analysis agent with the shared context
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    agent = create_pandas_dataframe_agent(
        llm,
        shared_context["df"],
        verbose=True,
        allow_dangerous_code=True
    )

    # Step 4: Clean and validate the DataFrame using the agent
    question = """
    You are tasked with cleaning the Event Log saved in the given dataframe.
    Only solve mistakes of the given variety
    Mistake type: Incorrect Position.
    This quality issue refers to the scenario where a case has been executed in reality but it has not been recorded in the log. 
    For example, assume that in the period between “01-01-2012” and “30-06-2012” 500 cases have been executed in reality. However, in the event log only 450 cases exist. 
    As another example, we might encounter an event log where we notice case ids missing in consecutive set of numbers, e.g., we might have an event log with case ids ..., case 888, case 889, case 891, case 892, .... It is most likely that case 890 is missing from this event log.
    """
    try:
        response = agent.invoke({"input": question})

        # Update the DataFrame in the shared context after processing
        shared_context["df"] = agent.tools[0].locals["df"]  # Retrieve updated DataFrame from tool's context
        logger.info(f"Agent Response:\n{response}")
        logger.info(f"Updated DataFrame shape: {shared_context['df'].shape}")

        # Save the updated DataFrame
        output_file_path = "./Pizza_Case_Real_cleaned_pres.csv"
        shared_context["df"].to_csv(output_file_path, index=False)
        logger.info(f"Cleaned DataFrame saved to {output_file_path}")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()



# standard
import os
import pandas as pd
import logging

# For getting the key
from dotenv import load_dotenv

# LangChain-related imports
from langchain_openai.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Mistake: all events with the activity name "Pizza received" have been deleted

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
    file_path = "./Pizza_Case_CSV_I14.csv"
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
    Can you spot the quality issue "Incorrect Position" in the given dataframe df?
    This quality issue is prevalent in logs that do not have timestamps associated with events. 
    Typically, in logs that do not have timestamps, the order in which the events appear is assumed to be the order in which they have been executed, i.e., the position of events in a case deﬁnes its execution order.
    It is possible that the mistake type does not exist or can't be found without more infromation.
    If possible Fix the Event Log Quality Issue.
    If the dataframe has been changed, Save the updated dataframe, as ./Pizza_Case_CSV_blind_I14.csv
    """

    try:
        response = agent.invoke(question)

        # The agent modifies its internal copy of the DataFrame; retrieve the updated copy:
        updated_df = agent.df

        logger.info(f"Agent response:\n{response}")
        logger.info(f"Updated DataFrame shape: {updated_df.shape}")

        # Save the updated DataFrame
        output_file_path = "./Pizza_Case_CSV_blind_I14.csv"
        updated_df.to_csv(output_file_path, index=False)
        logger.info(f"Cleaned DataFrame saved to {output_file_path}")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
