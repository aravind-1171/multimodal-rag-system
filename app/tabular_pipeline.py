import pandas as pd
import os

def extract_text_from_csv(file_path):
    """
    Reads a CSV file and converts it to
    readable text chunks for embedding
    """
    df = pd.read_csv(file_path)

    print(f"✅ CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"📊 Columns: {list(df.columns)}")

    chunks = []

    # Chunk 1 — overall summary
    summary = f"""Dataset Summary:
Total rows: {df.shape[0]}
Total columns: {df.shape[1]}
Columns: {', '.join(df.columns.tolist())}
Numeric columns: {', '.join(df.select_dtypes(include='number').columns.tolist())}
"""
    chunks.append(summary)

    # Chunk 2 — statistics for numeric columns
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        stats = numeric_df.describe().to_string()
        chunks.append(f"Statistical Summary:\n{stats}")

    # Chunk 3 — each row as readable text
    for _, row in df.iterrows():
        row_text = " | ".join(
            [f"{col}: {val}" for col, val in row.items()]
        )
        chunks.append(row_text)

    print(f"✅ CSV converted to {len(chunks)} text chunks")
    return chunks


# ── Test ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Create a sample CSV to test
    sample_data = """name,age,city,salary
Alice,28,Bangalore,85000
Bob,32,Mumbai,92000
Charlie,25,Chennai,78000
Diana,30,Hyderabad,95000
Eve,27,Kochi,82000
"""
    with open("sample.csv", "w") as f:
        f.write(sample_data)

    chunks = extract_text_from_csv("sample.csv")
    print("\n📄 Sample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:\n{chunk}")