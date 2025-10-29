import pickle
import argparse
import sys
import os


def extract_instances(input_file, output_file, num_instances, offset=0):
    """
    Extract n instances from a pickle file containing multiple instances.

    Args:
        input_file: Path to input pickle file
        output_file: Path to output pickle file
        num_instances: Number of instances to extract
        offset: Starting index for extraction (default: 0)
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    # Load original instances
    print(f"Loading instances from: {input_file}")
    with open(input_file, 'rb') as f:
        instances = pickle.load(f)

    total_instances = len(instances)
    print(f"Total instances in original file: {total_instances}")

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

    # Extract subset
    extracted_instances = instances[offset:end_index]

    # Save to new file
    print(f"\nExtracting instances {offset} to {end_index - 1} ({num_instances} total)...")
    with open(output_file, 'wb') as f:
        pickle.dump(extracted_instances, f)

    # Print statistics
    print(f"\nSuccessfully created: {output_file}")
    print(f"Number of instances extracted: {num_instances}")
    print(f"Instance range: [{offset}, {end_index - 1}]")
    print(f"\nInstance info:")
    print(f"  Type: {type(extracted_instances[0])}")
    if hasattr(extracted_instances[0], 'shape'):
        print(f"  Shape: {extracted_instances[0].shape}")
    elif hasattr(extracted_instances[0], '__len__'):
        print(f"  Length: {len(extracted_instances[0])}")
    else:
        print(f"  Value: {extracted_instances[0]}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract n instances from a pickle file containing multiple instances.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract first 10 instances
  python extract_n_instances.py -i instances/tsp/test/tsp100_test_small_seed1235.pkl -o instances/tsp/test/tsp100_10inst.pkl -n 10

  # Extract 5 instances starting from index 10
  python extract_n_instances.py -i instances/tsp/test/tsp100_test_small_seed1235.pkl -o instances/tsp/test/tsp100_5inst_offset10.pkl -n 5 --offset 10

  # Extract first 100 instances
  python extract_n_instances.py -i instances/tsp/test/tsp100_test_small_seed1235.pkl -o instances/tsp/test/tsp100_100inst.pkl -n 100
        """
    )

    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='Path to input pickle file containing instances')
    parser.add_argument('-o', '--output_file', type=str, required=True,
                        help='Path to output pickle file to create')
    parser.add_argument('-n', '--num_instances', type=int, required=True,
                        help='Number of instances to extract')
    parser.add_argument('--offset', type=int, default=0,
                        help='Starting index for extraction (default: 0)')

    args = parser.parse_args()

    # Extract instances
    extract_instances(
        input_file=args.input_file,
        output_file=args.output_file,
        num_instances=args.num_instances,
        offset=args.offset
    )


if __name__ == "__main__":
    main()
