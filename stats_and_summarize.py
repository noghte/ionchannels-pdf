import json

# Load the original JSON file
with open('./results/results-all-model_gpt-4o.json', 'r') as file:
    data = json.load(file)

# Initialize variables for statistics
query1_found_count = 0
query1_not_found_count = 0
query2_found_count = 0
query2_not_found_count = 0

# Initialize dictionary for summary data
summary_data = {}

# Process each entry in the JSON data
for uniprot_id, queries in data.items():
    # Initialize summary for this uniprot_id
    summary_data[uniprot_id] = {}

    # Process query1
    query1 = queries.get("query1", {})
    if query1:
        summary_data[uniprot_id]["query1"] = {
            "text": query1["text"],
            "answer": query1["answer"],
            "confidence": query1["confidence"],
            "evidence": query1["evidence"]
        }

        # Update statistics for query1
        if query1["answer"].lower() == "found":
            query1_found_count += 1
        else:
            query1_not_found_count += 1

    # Process query2
    query2 = queries.get("query2", {})
    if query2:
        summary_data[uniprot_id]["query2"] = {
            "text": query2["text"],
            "answer": query2["answer"],
            "confidence": query2["confidence"],
            "evidence": query2["evidence"]
        }
        # Update statistics for query1
        if query2["answer"].lower() == "found":
            query2_found_count += 1
        else:
            query2_not_found_count += 1

# Save the summary data to a new JSON file
with open('./results/results-summary.json', 'w') as summary_file:
    json.dump(summary_data, summary_file, indent=4)

# Print out the statistics
print(f"Statistics for query1:")
print(f"Found: {query1_found_count}")
print(f"Not Found: {query1_not_found_count}")
print(f"Statistics for query2:")
print(f"Found: {query2_found_count}")
print(f"Not Found: {query2_not_found_count}")