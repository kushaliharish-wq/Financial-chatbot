import re
import os
import pyodbc
from openai import OpenAI
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import datetime
import json
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from dotenv import load_dotenv

AZURE_SQL_CONNECTION_STRING = os.getenv("AZURE_SQL_CONNECTION_STRING")
engine = create_engine(AZURE_SQL_CONNECTION_STRING)

transactions = pd.read_csv("/personal_transactions.csv")
budgets = pd.read_csv("/Budget.csv")

transactions["Date"] = pd.to_datetime(transactions["Date"], format="%m/%d/%Y")
transactions["Category"] = transactions["Category"].str.strip().str.lower()
budgets["Category"] = budgets["Category"].str.strip().str.lower()


transactions.drop_duplicates(inplace=True)
budgets.drop_duplicates(subset="Category", inplace=True)

if "Budget" in budgets.columns:
    budgets["Budget"] = budgets["Budget"].fillna(0)  # Replace NaN budgets with 0
else:
    print("Error: 'Budget' column not found in 'budgets' DataFrame")


budgets["Category"] = budgets["Category"].fillna("misc")
transactions["Category"] = transactions["Category"].fillna("misc")

transactions["Account Name"] = transactions["Account Name"].fillna("unknown")
transactions["Transaction Type"] = transactions["Transaction Type"].fillna("unknown")

if budgets.isnull().any().any() or transactions.isnull().any().any():
    print("Warning: Some NaN values remain in the data after cleaning.")

transactions_df = transactions.rename(
    columns={
        "Date": "date",
        "Description": "description",
        "Amount": "amount",
        "Category": "transaction_category",
        "Account Name": "account_name",
        "Transaction Type": "transaction_type",
    }
)
budgets_df = budgets.rename(columns={"Category": "category", "Budget": "budget"})

transactions_df.to_sql("Transactions", engine, if_exists="append", index=False)
budgets_df.to_sql("Budgets", engine, if_exists="append", index=False)
print("Data successfully inserted into Azure SQL!")
