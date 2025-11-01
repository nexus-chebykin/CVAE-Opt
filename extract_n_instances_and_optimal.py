import pickle
import argparse
import sys
import os
import numpy as np


def extract_instances_and_solutions(input_instances_file, input_solutions_file, output_file, num_instances, offset=0):
    """
    Extract n instances and corresponding optimal solutions from pickle files.

    Args:
        input_instances_file: Path to input pickle file containing instances
        input_solutions_file: Path to input pickle file containing optimal solutions
        output_file: Path to output pickle file
        num_instances: Number of instances to extract
        offset: Starting index for extraction (default: 0)
    """
    # Check if input files exist
    if not os.path.exists(input_instances_file):
        print(f"Error: Input instances file '{input_instances_file}' not found.")
        sys.exit(1)

    if not os.path.exists(input_solutions_file):
        print(f"Error: Input solutions file '{input_solutions_file}' not found.")
        sys.exit(1)

    # Load original instances
    print(f"Loading instances from: {input_instances_file}")
    with open(input_instances_file, 'rb') as f:
        instances = pickle.load(f)

    # Load optimal solutions
    print(f"Loading solutions from: {input_solutions_file}")
    with open(input_solutions_file, 'rb') as f:
        solutions_data = pickle.load(f)

    # Handle conc-optimal100.pkl format: solutions_data is (list_of_tuples, something)
    # where each tuple is (cost, tour, time)
    if isinstance(solutions_data, tuple) and len(solutions_data) >= 2:
        # Format: ([( cost, tour, time), (cost, tour, time), ...], ...)
        if isinstance(solutions_data[0], list) and len(solutions_data[0]) > 0:
            if isinstance(solutions_data[0][0], tuple) and len(solutions_data[0][0]) >= 2:
                # Extract tours (index 1) from (cost, tour, time) tuples
                print("Detected conc-optimal format: extracting tours from (cost, tour, time) tuples...")
                solutions = [item[1] for item in solutions_data[0]]
            else:
                # Format: [tour, tour, ...]
                solutions = solutions_data[0]
        else:
            solutions = solutions_data[0]
    elif isinstance(solutions_data, list):
        # Format: [tour, tour, ...]
        solutions = solutions_data
    else:
        solutions = solutions_data

    total_instances = len(instances)
    total_solutions = len(solutions)

    print(f"Total instances in original file: {total_instances}")
    print(f"Total solutions in original file: {total_solutions}")

    # Validate that instances and solutions match
    if total_instances != total_solutions:
        print(f"Warning: Number of instances ({total_instances}) != number of solutions ({total_solutions})")
        print(f"Will use minimum of both: {min(total_instances, total_solutions)}")
        total_instances = min(total_instances, total_solutions)

    # Validate bounds
    if offset < 0:
        print(f"Error: Offset must be non-negative (got {offset}).")
        sys.exit(1)

    if offset >= total_instances:
        print(f"Error: Offset {offset} exceeds total instances {total_instances}.")
        sys.exit(1)

    if num_instances <= 0:
        print(f"Error: Number of instances must be positive (got {num_instances}).")
        sys.exit(1)

    end_index = offset + num_instances
    if end_index > total_instances:
        print(f"Warning: Requested {num_instances} instances from offset {offset}, but only {total_instances - offset} available.")
        print(f"Extracting {total_instances - offset} instances instead (indices {offset} to {total_instances - 1}).")
        end_index = total_instances
        num_instances = total_instances - offset

    # Extract subset from both instances and solutions
    extracted_instances = instances[offset:end_index]
    extracted_solutions = solutions[offset:end_index]

    # Save as tuple (instances, solutions)
    print(f"\nExtracting instances and solutions {offset} to {end_index - 1} ({num_instances} total)...")
    with open(output_file, 'wb') as f:
        pickle.dump((extracted_instances, extracted_solutions), f)

    # Print statistics
    print(f"\nSuccessfully created: {output_file}")
    print(f"Number of instances extracted: {num_instances}")
    print(f"Number of solutions extracted: {num_instances}")
    print(f"Range: [{offset}, {end_index - 1}]")
    print(f"\nFile format: (instances_list, solutions_list) tuple")
    print(f"\nInstance info:")
    print(f"  Type: {type(extracted_instances[0])}")
    if hasattr(extracted_instances[0], 'shape'):
        print(f"  Shape: {extracted_instances[0].shape}")
    elif hasattr(extracted_instances[0], '__len__'):
        print(f"  Length: {len(extracted_instances[0])} nodes")

    print(f"\nSolution info:")
    print(f"  Type: {type(extracted_solutions[0])}")
    if hasattr(extracted_solutions[0], '__len__'):
        print(f"  Length: {len(extracted_solutions[0])} nodes in tour")


def main():
    parser = argparse.ArgumentParser(
        description="Extract n instances and their optimal solutions from pickle files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract first 10 instances and solutions
  python extract_n_instances_and_optimal.py \\
    -i instances/tsp/test/tsp100_test_small_seed1235.pkl \\
    -s conc-optimal100.pkl \\
    -o instances/tsp/test/tsp100_10inst.pkl \\
    -n 10

  # Extract 5 instances starting from index 10
  python extract_n_instances_and_optimal.py \\
    -i instances/tsp/test/tsp100_test_small_seed1235.pkl \\
    -s conc-optimal100.pkl \\
    -o instances/tsp/test/tsp100_5inst_offset10.pkl \\
    -n 5 --offset 10
        """
    )

    parser.add_argument('-i', '--input_instances', type=str, required=True,
                        help='Path to input pickle file containing instances')
    parser.add_argument('-s', '--input_solutions', type=str, required=True,
                        help='Path to input pickle file containing optimal solutions')
    parser.add_argument('-o', '--output_file', type=str, required=True,
                        help='Path to output pickle file to create')
    parser.add_argument('-n', '--num_instances', type=int, required=True,
                        help='Number of instances to extract')
    parser.add_argument('--offset', type=int, default=0,
                        help='Starting index for extraction (default: 0)')

    args = parser.parse_args()

    # Extract instances and solutions
    extract_instances_and_solutions(
        input_instances_file=args.input_instances,
        input_solutions_file=args.input_solutions,
        output_file=args.output_file,
        num_instances=args.num_instances,
        offset=args.offset
    )


if __name__ == "__main__":
    main()
