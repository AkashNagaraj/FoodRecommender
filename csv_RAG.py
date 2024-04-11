import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain_community.llms import Ollama
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
import torch
from sqlalchemy import create_engine
import sys

torch.cuda.set_device(0)
    

#2) Get output as code
def script():
    model = Ollama(model="codellama")
    question = input("Enter your question :")
    # Implement llama-index to get exact rows/column names
    prompt = f"""
    You should return the python code given a query. Do not return an explaination of how you reached that answer. 
    
    Question:
        1. Write a function in python named get_data() which takes "df" as input. 
        2. {question} from the given input and return it. 
        3. Import libraries used in the code
        4. Indicate comment <Start> before the start of code and <End> after the code.
        5. Keep in mind this exact code should be runnable when copied to a .py file.

    Returns:
        str: only the python function
    """
    response = model.invoke(prompt)
    return(response)


def write_to_code(code):
    result = []
    code = code.split("\n")
    eliminate_chars = ["<",">","```"]
    for line in code:
        count = 0
        for char in eliminate_chars:
            if char in line:
                count = 1
        if count==0:
            result.append(line+"\n") 
    file = open("temp.py","w+")
    file.write("\nimport pandas as pd\n")
    file.writelines(result)
    file.write("""\ndf = pd.read_csv(r"data/clean_nutrients.csv")\n""")
    file.write("""print(get_data(df))""")
    file.close()
    

def run_code():
    from subprocess import call
    call(["python3", "temp.py"])


def main():
    response = script()
    write_to_code(response)
    run_code()


if __name__=="__main__":
    main()
