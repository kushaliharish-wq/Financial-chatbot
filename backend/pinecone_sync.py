import os
import numpy as np
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sqlalchemy import create_engine
from tqdm.auto import tqdm
import time
import getpass
from sentence_transformers import SentenceTransformer

pc = Pinecone(os.getenv(api_key="PINECONE_API_KEY"))
index_name = "final2-index"
dimension = 1920
existing_indexes = pc.list_indexes().names()
if index_name in existing_indexes:
    pc.delete_index(index_name)
pc.create_index(
    name=index_name,
    dimension=dimension,
    metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)
index = pc.Index(index_name)

AZURE_SQL_CONNECTION_STRING = "AZURE_SQL_CONNECTION_STRING"
engine = create_engine(AZURE_SQL_CONNECTION_STRING)


def fetch_data():
    query = """
    SELECT id, date, description, amount, transaction_category, account_name, transaction_type
    FROM dbo.Transactions
    """
    with engine.connect() as connection:
        data = pd.read_sql_query(query, connection)
    return data


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embedding(text):
    """Generate embeddings using SentenceTransformer."""
    vector = embedding_model.encode(text)
    return vector


def concatenate_embeddings(*embeddings, target_dim=1920):
    combined = []
    for emb in embeddings:
        combined.extend(emb)
    if len(combined) < target_dim:
        combined += [0] * (target_dim - len(combined))  # Padding
    else:
        combined = combined[:target_dim]  # Truncation
    return combined


def sync_with_pinecone(data, index):
    batch_size = 100
    total_batches = (len(data) + batch_size - 1) // batch_size
    for i in tqdm(
        range(0, len(data), batch_size),
        desc="Processing Batches",
        unit="batch",
        total=total_batches,
    ):
        i_end = min(len(data), i + batch_size)
        batch = data.iloc[i:i_end]
        ids = [str(row["id"]) for _, row in batch.iterrows()]
        # Generate embeddings for each column
        date_embeds = [
            generate_embedding(str(row["date"])) for _, row in batch.iterrows()
        ]
        print(f"Generated {len(date_embeds)} date embeddings")
        description_embeds = [
            generate_embedding(str(row["description"])) for _, row in batch.iterrows()
        ]
        print(f"Generated {len(description_embeds)} description embeddings")
        category_embeds = [
            generate_embedding(str(row["transaction_category"]))
            for _, row in batch.iterrows()
        ]
        print(f"Generated {len(category_embeds)} category embeddings")
        account_embeds = [
            generate_embedding(str(row["account_name"])) for _, row in batch.iterrows()
        ]
        print(f"Generated {len(account_embeds)} account embeddings")
        type_embeds = [
            generate_embedding(str(row["transaction_type"]))
            for _, row in batch.iterrows()
        ]
        print(f"Generated {len(type_embeds)} type embeddings")
        # Concatenate embeddings
        embeds = [
            concatenate_embeddings(date_emb, desc_emb, cat_emb, acc_emb, type_emb)
            for date_emb, desc_emb, cat_emb, acc_emb, type_emb in zip(
                date_embeds,
                description_embeds,
                category_embeds,
                account_embeds,
                type_embeds,
            )
        ]
        embeds = [np.array(embed, dtype=np.float64) for embed in embeds]
        metadata = [
            {
                "date": str(row["date"]),
                "description": row["description"],
                "amount": float(row["amount"]),
                "transaction_category": row["transaction_category"],
                "account_name": row["account_name"],
                "transaction_type": row["transaction_type"],
            }
            for _, row in batch.iterrows()
        ]
        with tqdm(
            total=len(ids), desc="Upserting Vectors", unit="vector"
        ) as upsert_pbar:
            index.upsert(vectors=zip(ids, embeds, metadata))
            upsert_pbar.update(len(ids))
        print(f"Upserted {len(ids)} vectors")


if __name__ == "__main__":
    data = fetch_data()
    sync_with_pinecone(data, index)
