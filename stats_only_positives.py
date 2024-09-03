import json

# Load the summary JSON file
with open('./results/results-all-model_gpt-4o.json', 'r') as file:
    data = json.load(file)

# Initialize a dictionary to store only the "Found" results
found_results = {}

# Iterate through each entry in the data
for uniprot_id, queries in data.items():
    # Initialize a dictionary to store found queries for this uniprot_id
    found_queries = {}
    
    # Check if query1 has the answer "Found"
    if "query1" in queries and queries["query1"]["answer"].lower() == "found":
        found_queries["query1"] = queries["query1"]
    
    # Check if query2 has the answer "Found"
    if "query2" in queries and queries["query2"]["answer"].lower() == "found":
        found_queries["query2"] = queries["query2"]
    
    # If there are any found queries, add them to the found_results
    if found_queries:
        found_results[uniprot_id] = found_queries

# Save the found results to a new JSON file
with open('results-found-full.json', 'w') as found_file:
    json.dump(found_results, found_file, indent=4)

print("Filtered results saved to results-found.json")
