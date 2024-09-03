import json
import pandas as pd

def process_data():
    # Load the JSON file
    with open('./results/results-all-model_gpt-4o.json', 'r') as file:
        data = json.load(file)

    # Initialize dictionaries to store results
    found_results = {}
    not_found_results = {}
    combined_results = {}
    summary_data = []

    # Iterate through each entry in the data
    for uniprot_id, queries in data.items():
        found_queries = {}
        not_found_queries = {}
        combined_queries = {}
        query1_found = query2_found = None
        confidence1 = confidence2 = None

        # Check and process query1
        if "query1" in queries:
            query1 = queries["query1"]
            answer1 = query1["answer"].lower() == "found"
            confidence1 = query1["confidence"]
            combined_queries["query1"] = query1

            if answer1:
                found_queries["query1"] = query1
                query1_found = True
            else:
                not_found_queries["query1"] = query1
                query1_found = False

        # Check and process query2
        if "query2" in queries:
            query2 = queries["query2"]
            answer2 = query2["answer"].lower() == "found"
            confidence2 = query2["confidence"]
            combined_queries["query2"] = query2

            if answer2:
                found_queries["query2"] = query2
                query2_found = True
            else:
                not_found_queries["query2"] = query2
                query2_found = False

        # Save found and not found queries to respective dictionaries
        if found_queries:
            found_results[uniprot_id] = found_queries
        if not_found_queries:
            not_found_results[uniprot_id] = not_found_queries

        # Save all queries to combined results
        if combined_queries:
            combined_results[uniprot_id] = combined_queries

        # Add summary data for this uniprot_id
        summary_data.append({
            "uniprot": uniprot_id,
            "answer1": query1_found,
            "confidence1": confidence1,
            "answer2": query2_found,
            "confidence2": confidence2
        })

    # Save the found results to a new JSON file
    with open('results-found.json', 'w') as found_file:
        json.dump(found_results, found_file, indent=4)

    # Save the not found results to a new JSON file
    with open('results-nofound.json', 'w') as not_found_file:
        json.dump(not_found_results, not_found_file, indent=4)

    # Save the combined results to a new JSON file
    with open('results.json', 'w') as combined_file:
        json.dump(combined_results, combined_file, indent=4)

    # Save the summary data to a CSV file
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv('results-summary.csv', index=False)

if __name__ == "__main__":
    process_data()

