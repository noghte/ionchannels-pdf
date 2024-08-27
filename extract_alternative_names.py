import pandas as pd
import requests

csv_path = './human_IC_annotation_sample.csv' #will be overwritten
def get_alternative_names_sparql(uniprot_ids):
    # Construct the VALUES clause with all UniProt IDs
    values_clause = " ".join([f"<{uid}>" for uid in uniprot_ids])
    
    # Construct the SPARQL query
    query = f"""
    BASE <http://purl.uniprot.org/uniprot/> 
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#> 
    PREFIX up: <http://purl.uniprot.org/core/> 

    SELECT ?protein ?altName 
    WHERE
    {{
        VALUES ?protein {{{values_clause}}}
        ?protein up:alternativeName ?altNameObj .
        ?altNameObj up:shortName ?altName .
    }}
    """
    
    # Define the SPARQL endpoint URL
    url = "https://sparql.uniprot.org/sparql"
    
    # Parameters for the request
    params = {
        'query': query,
        'format': 'json'
    }
    
    # Headers for the request
    headers = {
        'Accept': 'application/sparql-results+json'
    }
    
    # Make the request to the SPARQL endpoint
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        # Mapping from UniProt ID to alternative names
        alternative_names_dict = {}
        for result in data['results']['bindings']:
            protein = result['protein']['value'].split('/')[-1]
            alt_name = result['altName']['value']
            if protein in alternative_names_dict:
                alternative_names_dict[protein] += f", {alt_name}"
            else:
                alternative_names_dict[protein] = alt_name
        return alternative_names_dict
    else:
        return None

# Load the CSV file
df = pd.read_csv(csv_path)

# Extract the list of UniProt IDs from the DataFrame
uniprot_ids = df['Uniprot'].dropna().unique()

# Get alternative names using the SPARQL endpoint
alternative_names_dict = get_alternative_names_sparql(uniprot_ids)

# Update the AlternativeNames column in the DataFrame
if alternative_names_dict:
    for index, row in df.iterrows():
        if True:#pd.isna(row['AlternativeNames']) or row['AlternativeNames'] == '':
            uniprot_id = row['Uniprot']
            if uniprot_id in alternative_names_dict:
                df.at[index, 'AlternativeNames'] = alternative_names_dict[uniprot_id]

# Save the updated CSV file

df.to_csv(csv_path, index=False)
print(f"Updated CSV file saved as: {csv_path}")
