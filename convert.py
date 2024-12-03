# Open the .obo file and the output file for writing triples
input_file = "hp.obo"  # Path to your .obo file
output_file = "hp_anyburl_format.txt"  # Output file in triple format

# Initialize variables
term_id = None

# Read the .obo file line by line
with open(input_file, "r") as obo_file, open(output_file, "w") as out_file:
    for line in obo_file:
        line = line.strip()

        # Detect new terms
        if line.startswith("[Term]"):
            term_id = None  # Reset term_id for each new term

        # Extract the term ID
        elif line.startswith("id:") and not term_id:
            term_id = line.split(" ")[1]

        # Process name
        elif line.startswith("name:") and term_id:
            name = line[6:].strip()
            out_file.write(f"{term_id} has_name \"{name}\"\n")

        # Process definition
        elif line.startswith("def:") and term_id:
            definition = line[5:].split("\"")[1].strip()
            out_file.write(f"{term_id} has_definition \"{definition}\"\n")

        # Process synonyms
        elif line.startswith("synonym:") and term_id:
            synonym = line.split("\"")[1].strip()
            out_file.write(f"{term_id} synonym \"{synonym}\"\n")

        # Process xrefs
        elif line.startswith("xref:") and term_id:
            xref = line[6:].strip()
            out_file.write(f"{term_id} xref {xref}\n")

        # Process is_a relationships
        elif line.startswith("is_a:") and term_id:
            parent_id = line.split(" ")[1].strip()
            out_file.write(f"{term_id} is_a {parent_id}\n")

        # Add more fields if needed...

print(f"Conversion complete! Triples saved to {output_file}")
