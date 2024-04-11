from nltk.corpus import words, stopwords
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
tqdm.pandas()
from multiprocessing.pool import ThreadPool as Pool
import nltk
from nltk.tokenize import word_tokenize
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import sys
import os
import nltk
#nltk.download('words')
nltk.download('punkt')

os.environ["MOZ_HEADLESS"] = "1"

stop_words = set(stopwords.words("english"))


def remove_special_char(string):
    return "".join(ch if ch.isalnum() else " " for ch in string)


def scrape_price(data):

    top_k = 10
    input_query = " ".join(data)
    link = f"https://www.bigbasket.com/ps/?q={input_query}&nc=as"

    browser = webdriver.Firefox()
    browser.get(link)
    html = browser.page_source
    soup = BeautifulSoup(html, features="lxml")
    browser.close()
    product_name = [val.get_text() for val in soup.findAll("h3",{"class":"block m-0 line-clamp-2 font-regular text-base leading-sm text-darkOnyx-800 pt-0.5 h-full"})]
    quantity = [val.get_text() for val in soup.findAll("span",{"class":"Label-sc-15v1nk5-0 gJxZPQ truncate"})]
    prices = [val.get_text() for val in soup.findAll("span",{"class":"Label-sc-15v1nk5-0 Pricing___StyledLabel-sc-pldi2d-1 gJxZPQ AypOi"})] 
    
    top_k = min(len(product_name),len(quantity),len(prices),top_k) # Use only fixed number of items
    
    product_name = product_name[:top_k]
    quantity = quantity[:top_k]
    prices = prices[:top_k]

    assert len(product_name) == len(quantity) == len(prices), f"\nError in link: {link} \n Data:\n Product - {product_name}, quantity{quantity} and prices{prices}"

    result = []
    for idx in range(len(quantity)):
        if quantity[idx].split()[-1]=="g":
            weight = int(quantity[idx].split()[0])
            result.append([weight, product_name[idx], float(prices[idx].replace("₹",""))])
    df = pd.DataFrame(result, columns=["quantity","product_name","prices"])
    df["product_name"] = df["product_name"].apply(lambda x:re.sub(r'[^\w\s]','',x.lower()))
    df["product_name"] = df["product_name"].apply(word_tokenize).apply(lambda x:[word for word in x if word not in stop_words]).apply(lambda x: " ".join(x[:2]))
    return df


def get_prices(x):
    
    recipe = [[word for word in remove_special_char(phrase).split(" ") if word not in stop_words and word !=""][-3:] for phrase in x.split(",")]
    n_gram = 2
    quantity , product_name, prices = [],[],[]
    # Iterate over a two-word recipe 
    for item in tqdm(recipe):
        value = 0
        for idx in range(len(item)):
            price_df = scrape_price(item[idx:idx+n_gram])
            quantity += price_df["quantity"].tolist()
            product_name +=  price_df["product_name"].tolist()
            prices += price_df["prices"].tolist()

    return(quantity, product_name, prices)
    

def read_data():
    df = pd.read_csv("data/IndianFoodDatasetCSV.csv")
    df.dropna(inplace = True)
    return df


def save_data(df, path):
    with pd.ExcelWriter(path, engine="openpyxl",mode='a',if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
    print("Data saved")


def remove_non_english(text):
    english_words = set(words.words())
    tokens = word_tokenize(text)
    english_tokens = [word for word in tokens if word in english_words]
    return " ".join(english_tokens)

def make_dataset():
    df = read_data()
    pool = Pool(2)
    sample_size = 2
    
    df["RecipeName"] = df["RecipeName"].iloc[:sample_size].apply(lambda x: re.sub(r'[^\w\s]','',x.lower())).progress_apply(remove_non_english)
    
    print(df["URL"][:5])
    #1) Obtain prices of ingredients
    diet, quantity , product_name, prices, recipe = [], [], [], [], []
    # Iterate over all dishes
    ingredients = [(df["TranslatedIngredients"].iloc[idx]) for idx in df["TranslatedIngredients"].index[:sample_size]]
    result = pool.map(get_prices, ingredients)
    for idx,extracted_data in enumerate(result):
        quantity += extracted_data[0]
        product_name += extracted_data[1] 
        prices += extracted_data[2]
        recipe += [df["RecipeName"][idx]]*len(product_name)
        diet += [df["Diet"]]*len(product_name)
    
    all_prices_df = pd.DataFrame({"quantity(gm)":quantity, "product name":product_name, "prices":prices})
    recipes_and_ingredients = pd.DataFrame({"Recipe":recipe,"Product Name":recipe,"Diet":diet})
    
    print("Size of the dataframe: ",all_prices_df.shape) 
    
    prices_path = "data/food_prices.xlsx"
    recipe_path = "data/food_recipes.xlsx"
    save_data(all_prices_df, prices_path)
    save_data(recipes_and_ingredients, recipe_path)


def food_nutrients():
    nutrients_path = "data/nutrients/nutrition.csv"
    df = pd.read_csv(nutrients_path,index_col=0)
    #print("Columns with NaN:",df.isnull().any())
    df.drop(columns="saturated_fat",inplace=True)
    df["name"] = df["name"].apply(lambda x: x.lower())

    name = []
    english_words = set(words.words())
    for text in tqdm(df["name"].tolist()):
        english_tokens = []
        for word in word_tokenize(text): 
            if word in english_words:
                english_tokens.append(word)    
        if len(english_tokens)==0:
            name.append(np.nan)
        else:
            name.append(" ".join(english_tokens))
    df["name"] = name 
    df.dropna(inplace=True) 
    
    df.to_csv("data/clean_nutrients.csv")
    print("Saved Data")


def main():
    # 1) Dataset with recipes, ingredients, quantity and their prices
    #make_dataset()
    # 2) Dataset with food nutrients
    #food_nutrients()
    # 3) Get a user input
    # Extract relevant content using MedPaLM(Medical LLM)
    # Use our database for RAG
    # 4) Output
    # Available recipes, cost of recipes
    # 5) Monitor data statistics
    

if __name__=="__main__":
    main()
