#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.common.visualizer import plot_trial, compare_trials

def main():
    parser = argparse.ArgumentParser(description='Visualize logged trial data')
    parser.add_argument('trial_path', type=str, help='Path to trial directory (e.g., log/test_sim/)')
    parser.add_argument('--show', action='store_true', help='Display plots interactively (default: save only)')
    parser.add_argument('--bundle', type=str, default=None, help='Plot specific bundle only')
    parser.add_argument('--compare', nargs='+', type=str, default=None, 
                       help='Compare multiple trials (provide multiple paths)')
    parser.add_argument('--signal', type=str, default='q',
                       help='Signal name for comparison (default: q)')
    
    args = parser.parse_args()
    
    if args.compare:
        print(f"Comparing {len(args.compare)} trials on signal '{args.signal}'")
        fig = compare_trials(args.compare, signal_name=args.signal, show=args.show)
        print("Comparison complete")
    else:
        if args.bundle:
            from src.common.visualizer import Visualizer
            viz = Visualizer(args.trial_path)
            viz.plot_bundle(args.bundle, save=True)
            if args.show:
                viz.show_all()
            else:
                viz.close_all()
        else:
            plot_trial(args.trial_path, show=args.show)

if __name__ == "__main__":
    main()
