import random

# Input file with all triples
input_file = "/home/sulimaas/Anyburl/hp_anyburl_format.txt"
train_file = "/home/sulimaas/Anyburl/train.txt"
valid_file = "/home/sulimaas/Anyburl/valid.txt"
test_file = "/home/sulimaas/Anyburl/test.txt"

# Read the triples from the input file
with open(input_file, "r") as infile:
    triples = infile.readlines()

# Shuffle the triples randomly for unbiased splitting
random.shuffle(triples)

# Split data: 80% train, 10% valid, 10% test
train_split = int(0.8 * len(triples))
valid_split = int(0.9 * len(triples))

train_data = triples[:train_split]
valid_data = triples[train_split:valid_split]
test_data = triples[valid_split:]

# Write the splits to their respective files
with open(train_file, "w") as train_out:
    train_out.writelines(train_data)

with open(valid_file, "w") as valid_out:
    valid_out.writelines(valid_data)

with open(test_file, "w") as test_out:
    test_out.writelines(test_data)

print("Data has been split into train.txt, valid.txt, and test.txt")
