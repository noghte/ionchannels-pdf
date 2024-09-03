import re
import pandas as pd
import json
import matplotlib.pyplot as plt

def extract_pubmed_ids(pubmed_ids_str):
    if not isinstance(pubmed_ids_str, str):
        return []

    pattern = r'PMID:\s*(\d+)'
    matches = re.findall(pattern, pubmed_ids_str)
    pubmed_ids = [match for match in matches if match.isdigit()]

    return pubmed_ids if pubmed_ids else []

def extract_json_pubmed_ids(data):
    json_pubmed_ids = {}
    for uniprot_id, queries in data.items():
        pubmed_ids = []
        for query in queries.values():
            pubmed_ids.extend([source['pubmed_id'] for source in query['sources']])
        json_pubmed_ids[uniprot_id] = pubmed_ids
    return json_pubmed_ids

def match_pubmed_ids(csv_df, json_pubmed_ids):
    position_counts = {i: 0 for i in range(10)}  # Assuming a max of 10 sources per query
    unused_pubmed_ids = {}  # Dictionary to store Uniprot ID and unused PubMed IDs

    for uniprot_id, json_ids in json_pubmed_ids.items():
        csv_row = csv_df[csv_df['Uniprot'] == uniprot_id]
        if not csv_row.empty:
            csv_ids = csv_row.iloc[0]['ExtractedPubMedIDs']
            for csv_id in csv_ids:
                if csv_id in json_ids:
                    pos = json_ids.index(csv_id)
                    position_counts[pos] += 1
                else:
                    if uniprot_id not in unused_pubmed_ids:
                        unused_pubmed_ids[uniprot_id] = []
                    unused_pubmed_ids[uniprot_id].append(csv_id)  # Store unused PubMed ID

    return position_counts, unused_pubmed_ids

if __name__ == "__main__":
    # Load the CSV file
    csv_file = 'human_IC_annotation.csv'  # Replace with your actual CSV file path
    df = pd.read_csv(csv_file, nrows=259)

    # Extract PubMed IDs for each row in the CSV
    df['ExtractedPubMedIDs'] = df['PubMed'].apply(extract_pubmed_ids)
    
    with open('./results/results-all-model_gpt-4o.json', 'r') as file:
        json_data = json.load(file)
    
    # Extract PubMed IDs from JSON
    json_pubmed_ids = extract_json_pubmed_ids(json_data)

    # Analyze the positions of the matching PubMed IDs and collect unused IDs
    position_counts, unused_pubmed_ids = match_pubmed_ids(df, json_pubmed_ids)
    print("Position Counts:", position_counts)
    print(f"Number of PubMed IDs in CSV but not in JSON: {sum(len(ids) for ids in unused_pubmed_ids.values())}")
    print("Unused PubMed IDs by Uniprot ID:")
    for uniprot_id, ids in unused_pubmed_ids.items():
        print(f"{uniprot_id}: {ids}")

    # Plotting the position counts (without unused PubMed IDs)
    plt.figure(figsize=(10, 6))
    plt.bar(position_counts.keys(), position_counts.values(), color='skyblue', edgecolor='black')
    plt.xlabel('Position in JSON Sources List')
    plt.ylabel('Count of Matching PubMed IDs')
    plt.title('Count of CSV PubMed IDs Matching at Each Position in JSON')
    plt.xticks(range(len(position_counts)), labels=[f'Position {i+1}' for i in range(len(position_counts))], rotation=90)

    plt.tight_layout()  # Adjust layout to accommodate vertical labels
    plt.savefig('pubmed_position_distribution.png')
