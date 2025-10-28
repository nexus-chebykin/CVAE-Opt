import pickle
import sys

# Save just the first instance from the original file into a new file
with open('instances/tsp/test/tsp100_test_small_seed1235.pkl', 'rb') as f:
    instances = pickle.load(f)

print(f"Total instances in original file: {len(instances)}")

single_instance = [instances[0]]

with open('instances/tsp/test/tsp100_single.pkl', 'wb') as f:
    pickle.dump(single_instance, f)

print("Created tsp100_single.pkl with 1 instance")
print(f"Instance shape: {instances[0].shape}")
