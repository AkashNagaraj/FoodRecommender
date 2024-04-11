import torch
from langchain_community.llms import Ollama
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma


def read_data(question):
    loader = CSVLoader("data/nutrients/nutrition.csv")
    data = loader.load()
    data = data[:100]

    embedding_function = GPT4AllEmbeddings() # SentenceTransformersEmbeddings(model_name="")
    vectorstore = Chroma.from_documents(documents=data, embedding=embedding_function)

    print("Performing similarity search")
    docs = vectorstore.similarity_search(question)
    return docs


torch.cuda.set_device(0)


#2) Get output as code
def script(question):
    model = Ollama(model="codellama")
    # Implement llama-index to get exact rows/column names
    prompt = f"""
    Input Sentence - Suggest high protein food.
    Instruction - Given this sentence return the tokenized words as output in l.  
    """

    response = model.invoke(prompt)
    return(response)


def extract_data(docs):
    first_response = docs[0].dict()["page_content"].split("\n")
    result = {}
    row = first_response[0].replace(":","")

    for item in first_response[1:]:
        key,val = item.split(":")
        result[key] = val.strip()
    
    return result


def main_query(question):
    words = script(question) 
    print(words)
    sys.exit()

    docs = read_data(question)
    response = extract_data(docs)
    return response


if __name__=="__main__":
    _ = main_query(input("Enter query: "))
