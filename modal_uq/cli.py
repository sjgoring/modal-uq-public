
import argparse
from .runners.run_experiment import run_from_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to JSON config')
    args = ap.parse_args()
    run_from_config(args.config)

if __name__ == '__main__':
    main()
