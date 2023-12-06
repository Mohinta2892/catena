import os
import argparse


def create_directory_structure(base_dir, domains):
    # Create the base data directory
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Convert domain names to uppercase
    domains = [domain.upper() for domain in domains]

    # Create directories for each domain and data type
    for domain in domains:
        domain_dir = os.path.join(base_dir, domain)
        if not os.path.exists(domain_dir):
            os.makedirs(domain_dir)

        data_types = ['data_2d', 'data_3d']

        for data_type in data_types:
            data_type_dir = os.path.join(domain_dir, data_type)
            if not os.path.exists(data_type_dir):
                os.makedirs(data_type_dir)

            splits = ['train', 'test']

            for split in splits:
                split_dir = os.path.join(data_type_dir, split)
                if not os.path.exists(split_dir):
                    os.makedirs(split_dir)

    # Create the 'preprocessed' directory
    preprocessed_dir = os.path.join(base_dir, 'preprocessed')
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a directory structure for data")
    parser.add_argument("base_dir", help="Base directory for the data structure")
    parser.add_argument("domains", nargs="+", help="List of domain names (case-insensitive)")

    # Add help note
    note = "This script creates a directory structure for organizing data by domain, data type, and split.\n"
    note += "Domain names are case-insensitive, and they will be converted to uppercase."

    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = note

    args = parser.parse_args()

    create_directory_structure(args.base_dir, args.domains)
