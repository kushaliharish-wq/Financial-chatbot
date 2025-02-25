import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
import json
from datetime import datetime
import uvicorn
import google.generativeai as genai

api_key = "your_open_ai_key"
client = OpenAI(api_key=api_key)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key="your_pinecone_key")
index_name = "final2-index"
index = pc.Index(index_name)

genai.configure(api_key="your_google_api_key")
model = genai.GenerativeModel("gemini-pro")


def generate_enhanced_response(transaction_data: str, user_question: str) -> str:
    if not transaction_data or "No transaction found" in transaction_data:
        return "I couldn't find any matching transactions for your query. Please check the date and category and try again."

    prompt = f"""
    User Question: {user_question}
    Transaction Data: {transaction_data}
    
    Rules:
    1. If transaction data shows specific transaction details, format response as:
       "On [date], you spent $[amount] for [category] ([description])"
    2. If transaction data shows aggregate spending, format response as:
       "In [date], you spent a total of $[amount] on [category] across [number] transactions"
    3. If transaction data shows savings information, format response as:
       "In [date], you saved $[amount]. Your income was $[income] and your spending was $[spending]."
    4. Always include the exact amounts from the transaction data
    5. Be precise and direct in answering the amount spent or saved
    
    Please provide a clear, direct response focusing on the amount spent or saved.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a financial assistant."},
                {"role": "user", "content": prompt},
            ],
            model="gpt-3.5-turbo",
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return transaction_data


def extract_entities_from_query(query):
    prompt = f"""
    User query: "{query}"
    
    Extract the following details:
    - transaction_category: Category of spending (e.g., shopping, groceries,alcohol & bars,auto insurance,coffee shops,electronics & software, entertainment,fast food,gas & fuel,haircut,home improvement, internet, mobile phone,mortgage & rent,movies & dvds, 
music,restaurants,shopping,television,utilities)
    - description: Store or vendor name (like Thai Restaurant, Amazon, Netflix,Credit Card Payment,Mortgage Payment,Hardware Store,Phone Company,Grocery Store, 
City Water Charges,Biweekly Paycheck,Internet Service Provider,Brunch Restaurant,Barbershop,Brewing Company,BP, 
Movie Theater)
    - date_range: Specific date or range (YYYY-MM-DD or YYYY-MM)
    - amount: If the query asks for a specific amount

    Rules for EXACT category matching:
    1. If the word "internet" appears in the query, ALWAYS use "internet" as the category
    2. Categories must match EXACTLY one of these (do not substitute or combine):
       - shopping
       - groceries
       - alcohol & bars
       - auto insurance
       - coffee shops
       - electronics & software
       - entertainment
       - fast food
       - gas & fuel
       - haircut
       - home improvement
       - internet
       - mobile phone
       - mortgage & rent
       - movies & dvds
       - music
       - restaurants
       - utilities
    
    Rules for dates:
    1. For exact date queries (e.g., "2018-03-26"), set query_type as "single transaction"
    2. For month queries (e.g., "March 2018"), set query_type as "aggregate"
    3. For savings queries (e.g., "how much did I save in January 2018?"), set query_type as "savings"
    4. transaction_category must be lowercase and one of: shopping, groceries, restaurants, utilities
    5. query_type must be exactly "single", "aggregate", or "savings"
    6. If the query asks about savings, set is_savings_query to true

    Format as JSON:
    {{
        "transaction_category": "string",
        "description": "string",
        "date_range": "string",
        "amount": "float",
        "query_type": "string",
        "is_savings_query": "boolean"
    }}
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a financial assistant."},
                {"role": "user", "content": prompt},
            ],
            model="gpt-3.5-turbo",
        )

        response_content = chat_completion.choices[0].message.content

        if response_content.startswith("```json"):
            response_content = response_content[7:].strip()
        elif response_content.startswith("```"):
            response_content = response_content[3:].strip()

        if response_content.endswith("```"):
            response_content = response_content[:-3].strip()
        response_content = response_content.rstrip("`")

        print(f"Cleaned response: {response_content}")

        entities = json.loads(response_content)
        return entities

    except json.JSONDecodeError as e:
        print(f"Error parsing response as JSON: {e}")
        print(f"Response content was: {response_content}")
        # Return default values
        return {
            "transaction_category": "",
            "description": "",
            "date_range": "",
            "amount": 0,
            "query_type": "unknown",
        }


def search_pinecone(query):
    entities = extract_entities_from_query(query)
    query_type = entities.get("query_type", "")
    date = entities.get("date_range", entities.get("date", ""))
    print(f"Extracted date: {date}")
    if query_type == "single transaction":
        category = entities.get("transaction_category", "").lower()
        category_embedding = embed_model.encode(f"category: {category}")
        date_embedding = embed_model.encode(f"date: {date}")
        query_embedding = concatenate_embeddings(date_embedding, category_embedding)

        search_results = index.query(
            vector=query_embedding.tolist(),
            top_k=5,
            include_metadata=True,
        )

        return format_single_transaction(search_results, category, date)

    elif query_type == "aggregate":
        category = entities.get("transaction_category", "").lower()
        print(f"Extracted category: {category}")  # Debug print

        if not date:
            return "Please specify a date in YYYY-MM format."

        # Ensure date is in YYYY-MM format
        date = date[:7] if len(date) > 7 else date

        category_embedding = embed_model.encode(f"category: {category}")
        date_embedding = embed_model.encode(f"date: {date}")
        query_embedding = concatenate_embeddings(date_embedding, category_embedding)

        search_results = index.query(
            vector=query_embedding.tolist(), top_k=100, include_metadata=True
        )

        return format_aggregate_transactions(search_results, date, category)
    elif query_type == "savings":
        if not date:
            return "Please specify a date in YYYY-MM format."

        try:
            if date and not date.startswith("20"):
                print(f"Converting date from '{date}'")
                date_obj = datetime.strptime(date, "%B %Y")
                date = date_obj.strftime("%Y-%m")
                print(f"Converted to '{date}'")
        except ValueError as e:

            date = date[:7] if len(date) > 7 else date

        date_embedding = embed_model.encode(f"date: {date}")
        paycheck_embedding = embed_model.encode("category: paycheck type: credit")
        debit_embedding = embed_model.encode("type: debit")

        income_query_embedding = concatenate_embeddings(
            date_embedding, paycheck_embedding
        )
        credit_results = index.query(
            vector=income_query_embedding.tolist(), top_k=100, include_metadata=True
        )

        spending_query_embedding = concatenate_embeddings(
            date_embedding, debit_embedding
        )
        debit_results = index.query(
            vector=spending_query_embedding.tolist(), top_k=100, include_metadata=True
        )

        return calculate_savings(credit_results, debit_results, date)

    return "Could not determine the query type. Please try again."


def concatenate_embeddings(*embeddings, target_dim=1920):
    """Combine embeddings with proper dimensionality."""
    combined = []
    for emb in embeddings:
        combined.extend(emb)

    if len(combined) < target_dim:
        combined.extend([0] * (target_dim - len(combined)))
    return np.array(combined[:target_dim], dtype=np.float64)


def format_single_transaction(results, category, date):

    print(f"\nSearching for {category} transaction on {date}")

    if not results["matches"]:
        return "No matching transaction found."

    print("\nExamining matches:")
    matched_transactions = []

    for match in results["matches"]:
        metadata = match["metadata"]
        print(f"\nChecking transaction:")
        print(f"Date: {metadata['date']}")
        print(f"Category: {metadata['transaction_category'].lower()}")
        print(f"Amount: ${float(metadata['amount']):.2f}")
        print(f"Score: {match['score']}")
        exact_date_match = metadata["date"] == date
        category_match = metadata["transaction_category"].lower() == category
        # For single transactions, match exact date and category
        print(f"Date match: {exact_date_match}")
        print(f"Category match: {category_match}")

        if exact_date_match and category_match:
            print("✓ MATCH")
            matched_transactions.append((match["score"], metadata))
        else:
            print("✗ NO MATCH")
            if not exact_date_match:
                print(
                    f"  Reason: Date mismatch (expected {date}, got {metadata['date']})"
                )
            if not category_match:
                print(
                    f"  Reason: Category mismatch (expected {category}, got {metadata['transaction_category'].lower()})"
                )

    if not matched_transactions:
        return f"No transaction found for {category} on {date}."

    # Get the highest scoring match
    best_match = max(matched_transactions, key=lambda x: x[0])
    metadata = best_match[1]

    return f"""
    **Transaction Details:**
    - **Date:** {metadata['date']}
    - **Description:** {metadata['description']}
    - **Amount:** ${float(metadata['amount']):.2f}
    - **Category:** {metadata['transaction_category']}
    - **Account:** {metadata.get('account_name', 'N/A')}
    """


def format_aggregate_transactions(results, date, category):

    matching_transactions = []
    total_spent = 0.0

    print("\nAll transactions being considered:")
    for match in results["matches"]:
        metadata = match["metadata"]
        transaction_date = metadata["date"][:7]  # Get just YYYY-MM part
        matches_date = transaction_date == date
        matches_category = metadata["transaction_category"].lower() == category

        if matches_date and matches_category:
            print("✓ MATCH - Transaction included in calculation")
            matching_transactions.append(metadata)
            total_spent += float(metadata["amount"])
        else:
            print("✗ NO MATCH - Transaction excluded")
            if not matches_date:
                print(
                    f"  Reason: Date mismatch (expected {date}, got {transaction_date})"
                )
            if not matches_category:
                print(
                    f"  Reason: Category mismatch (expected {category}, got {metadata['transaction_category'].lower()})"
                )

    if not matching_transactions:
        return f"No transactions found for category '{category}' in {date}."

    # Sort transactions by date
    matching_transactions.sort(key=lambda x: x["date"])

    transactions_list = "\n".join(
        [
            f"- {tx['date']}: {tx['description']} (${float(tx['amount']):.2f})"
            for tx in matching_transactions
        ]
    )

    summary = f"""
    **Spending Summary for {category.title()} in {date}:**
    - **Total Amount:** ${total_spent:.2f}
    - **Number of Transactions:** {len(matching_transactions)}

    **Individual Transactions:**
    {transactions_list}
    """
    return summary


def calculate_savings(credit_results, debit_results, date):

    income_transactions = []

    for match in credit_results["matches"]:
        metadata = match["metadata"]
        transaction_date = metadata["date"][:7]
        transaction_type = metadata.get("transaction_type", "").lower()
        transaction_category = metadata.get("transaction_category", "").lower()

        print(
            f"Transaction: Date={transaction_date}, Type={transaction_type}, Category={transaction_category}, Amount=${float(metadata.get('amount', 0)):.2f}"
        )

        is_income = transaction_category == "paycheck" and transaction_type == "credit"

        if transaction_date == date and is_income:
            print(f"INCOME MATCH - Adding to income")
            income_transactions.append(metadata)
            total_income += float(metadata["amount"])
        else:
            print(f"NOT INCOME")
            reasons = []
            if transaction_date != date:
                reasons.append(
                    f"Date mismatch: expected {date}, got {transaction_date}"
                )
            if not is_income:
                reasons.append(f"Not a paycheck or not a credit transaction")
            for reason in reasons:
                print(f"  {reason}")
    spending_transactions = []
    total_spending = 0.0
    for match in debit_results["matches"]:
        metadata = match["metadata"]
        transaction_date = metadata["date"][:7]
        transaction_type = metadata.get("transaction_type", "").lower()

        print(
            f"Transaction: Date={transaction_date}, Type={transaction_type}, Amount=${float(metadata.get('amount', 0)):.2f}"
        )

        if transaction_date == date and transaction_type == "debit":
            print(f"✓ SPENDING MATCH - Adding to spending")
            spending_transactions.append(metadata)
            total_spending += float(metadata["amount"])
        else:
            print(f" NOT SPENDING")
            if transaction_date != date:
                print(f"  Date mismatch: expected {date}, got {transaction_date}")
            if transaction_type != "debit":
                print(f"  Type mismatch: expected debit, got {transaction_type}")

    savings = total_income - total_spending

    try:
        date_obj = datetime.strptime(date, "%Y-%m")
        formatted_date = date_obj.strftime("%B %Y")
    except:
        formatted_date = date

    # Generate summary
    income_list = "\n".join(
        [
            f"- {tx['date']}: {tx['description']} (${float(tx['amount']):.2f})"
            for tx in income_transactions[:5]  # Show only top 5 for brevity
        ]
    )

    spending_list = "\n".join(
        [
            f"- {tx['date']}: {tx['description']} (${float(tx['amount']):.2f})"
            for tx in spending_transactions[:5]  # Show only top 5 for brevity
        ]
    )

    summary = f"""
    **Savings Summary for {formatted_date}:**
    - **Total Income:** ${total_income:.2f} ({len(income_transactions)} transactions)
    - **Total Spending:** ${total_spending:.2f} ({len(spending_transactions)} transactions)
    - **Total Savings:** ${savings:.2f}
    
    **Income Transactions:**
    {income_list}
    {"..." if len(income_transactions) > 5 else ""}
    
    **Top Spending Transactions:**
    {spending_list}
    {"..." if len(spending_transactions) > 5 else ""}
    """
    return summary


app = FastAPI()


class QueryRequest(BaseModel):
    user_id: str
    question: str
    chat_history: list[str]


chat_history = {}


@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        transaction_data = search_pinecone(request.question)
        print(transaction_data)

        response = generate_enhanced_response(transaction_data, request.question)
        print(f"\nGenerated response: {response}")
        return {"response": response, "transaction_data": transaction_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8001)
