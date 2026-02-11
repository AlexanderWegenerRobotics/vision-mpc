import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

class Visualizer:
    def __init__(self, trial_path):
        self.trial_path = Path(trial_path)
        
        if not self.trial_path.exists():
            raise FileNotFoundError(f"Trial path not found: {trial_path}")
        
        metadata_path = self.trial_path / 'metadata.yaml'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = yaml.safe_load(f)
        else:
            self.metadata = {}
        
        self.figures = []
    
    def plot_bundle(self, bundle_name, save=True):
        from src.common.data_logger import DataLogger
        
        try:
            data = DataLogger.load(self.trial_path, bundle_name)
        except FileNotFoundError:
            print(f"Bundle '{bundle_name}' not found in trial")
            return
        
        timestamps = data['timestamps']
        t_relative = timestamps - timestamps[0]
        
        signals = {k: v for k, v in data.items() if k != 'timestamps'}
        
        n_signals = len(signals)
        fig, axes = plt.subplots(n_signals, 1, figsize=(12, 3*n_signals))
        if n_signals == 1:
            axes = [axes]
        
        for idx, (signal_name, signal_data) in enumerate(signals.items()):
            ax = axes[idx]
            
            if signal_data.ndim == 1:
                ax.plot(t_relative, signal_data, linewidth=1.5)
                ax.set_ylabel(signal_name)
            elif signal_data.ndim == 2:
                n_dims = signal_data.shape[1]
                for i in range(n_dims):
                    ax.plot(t_relative, signal_data[:, i], label=f'{signal_name}[{i}]', linewidth=1.0)
                ax.legend(loc='upper right', ncol=min(n_dims, 4))
                ax.set_ylabel(signal_name)
            else:
                ax.text(0.5, 0.5, f'{signal_name}: {signal_data.shape}\n(too high-dimensional to plot)', 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time [s]')
        
        trial_name = self.metadata.get('trial_name', 'unknown')
        fig.suptitle(f"Trial: {trial_name} | Bundle: {bundle_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            figures_dir = self.trial_path / 'figures'
            figures_dir.mkdir(exist_ok=True)
            filepath = figures_dir / f'{bundle_name}.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        self.figures.append(fig)
        return fig
    
    def plot_all_bundles(self, save=True):
        schemas = self.metadata.get('schemas', {})
        
        if not schemas:
            print("No bundles found in metadata")
            h5_files = list(self.trial_path.glob('*.h5'))
            bundle_names = [f.stem for f in h5_files]
        else:
            bundle_names = list(schemas.keys())
        
        print(f"Found {len(bundle_names)} bundles: {bundle_names}")
        
        for bundle_name in bundle_names:
            print(f"\nPlotting bundle: {bundle_name}")
            self.plot_bundle(bundle_name, save=save)
        
        return self.figures
    
    def plot_joint_trajectories(self, bundle_name='physics', save=True):
        from src.common.data_logger import DataLogger
        
        try:
            data = DataLogger.load(self.trial_path, bundle_name)
        except FileNotFoundError:
            print(f"Bundle '{bundle_name}' not found")
            return
        
        if 'q' not in data:
            print(f"No joint positions 'q' found in bundle '{bundle_name}'")
            return
        
        timestamps = data['timestamps']
        t_relative = timestamps - timestamps[0]
        q = data['q']
        qd = data.get('qd', None)
        
        n_joints = q.shape[1]
        
        fig, axes = plt.subplots(2 if qd is not None else 1, 1, figsize=(12, 8))
        if qd is None:
            axes = [axes]
        
        for i in range(n_joints):
            axes[0].plot(t_relative, q[:, i], label=f'q{i+1}', linewidth=1.5)
        axes[0].set_ylabel('Joint Position [rad]')
        axes[0].legend(loc='upper right', ncol=4)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlabel('Time [s]')
        
        if qd is not None:
            for i in range(n_joints):
                axes[1].plot(t_relative, qd[:, i], label=f'qd{i+1}', linewidth=1.5)
            axes[1].set_ylabel('Joint Velocity [rad/s]')
            axes[1].legend(loc='upper right', ncol=4)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlabel('Time [s]')
        
        trial_name = self.metadata.get('trial_name', 'unknown')
        fig.suptitle(f"Joint Trajectories: {trial_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            figures_dir = self.trial_path / 'figures'
            figures_dir.mkdir(exist_ok=True)
            filepath = figures_dir / 'joint_trajectories.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        self.figures.append(fig)
        return fig
    
    def plot_box_trajectory(self, bundle_name='physics', save=True):
        from src.common.data_logger import DataLogger
        
        try:
            data = DataLogger.load(self.trial_path, bundle_name)
        except FileNotFoundError:
            print(f"Bundle '{bundle_name}' not found")
            return
        
        if 'box_pos' not in data:
            print(f"No box position 'box_pos' found in bundle '{bundle_name}'")
            return
        
        timestamps = data['timestamps']
        t_relative = timestamps - timestamps[0]
        box_pos = data['box_pos']
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(box_pos[:, 0], box_pos[:, 1], linewidth=2, color='red')
        axes[0].scatter(box_pos[0, 0], box_pos[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
        axes[0].scatter(box_pos[-1, 0], box_pos[-1, 1], c='blue', s=100, marker='x', label='End', zorder=5)
        axes[0].set_xlabel('X [m]')
        axes[0].set_ylabel('Y [m]')
        axes[0].set_title('Box Trajectory (Top View)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        
        for i, label in enumerate(['X', 'Y', 'Z']):
            axes[1].plot(t_relative, box_pos[:, i], label=label, linewidth=1.5)
        axes[1].set_xlabel('Time [s]')
        axes[1].set_ylabel('Position [m]')
        axes[1].set_title('Box Position vs Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        trial_name = self.metadata.get('trial_name', 'unknown')
        fig.suptitle(f"Box Trajectory: {trial_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            figures_dir = self.trial_path / 'figures'
            figures_dir.mkdir(exist_ok=True)
            filepath = figures_dir / 'box_trajectory.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        self.figures.append(fig)
        return fig
    
    def show_all(self):
        plt.show()
    
    def close_all(self):
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()


def plot_trial(trial_path, show=False):
    viz = Visualizer(trial_path)
    
    print(f"\n{'='*60}")
    print(f"Visualizing trial: {trial_path}")
    print(f"{'='*60}\n")
    
    if viz.metadata:
        print("Metadata:")
        print(f"  Trial name: {viz.metadata.get('trial_name', 'N/A')}")
        print(f"  Duration: {viz.metadata.get('duration', 0):.2f} seconds")
        print(f"  Bundles: {list(viz.metadata.get('schemas', {}).keys())}")
        print()
    
    viz.plot_all_bundles(save=True)
    
    viz.plot_joint_trajectories(save=True)
    
    viz.plot_box_trajectory(save=True)
    
    print(f"\n{'='*60}")
    print(f"Generated {len(viz.figures)} figures")
    print(f"Saved to: {viz.trial_path / 'figures'}")
    print(f"{'='*60}\n")
    
    if show:
        viz.show_all()
    else:
        viz.close_all()
    
    return viz


def compare_trials(trial_paths, signal_name, bundle_name='physics', show=False):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for trial_path in trial_paths:
        from src.common.data_logger import DataLogger
        trial_path = Path(trial_path)
        
        try:
            data = DataLogger.load(trial_path, bundle_name)
        except FileNotFoundError:
            print(f"Bundle '{bundle_name}' not found in {trial_path}")
            continue
        
        if signal_name not in data:
            print(f"Signal '{signal_name}' not found in bundle '{bundle_name}' of {trial_path}")
            continue
        
        timestamps = data['timestamps']
        t_relative = timestamps - timestamps[0]
        signal_data = data[signal_name]
        
        trial_name = trial_path.name
        
        if signal_data.ndim == 1:
            ax.plot(t_relative, signal_data, label=trial_name, linewidth=1.5)
        elif signal_data.ndim == 2:
            mean_signal = np.mean(signal_data, axis=1)
            ax.plot(t_relative, mean_signal, label=f'{trial_name} (mean)', linewidth=1.5)
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(signal_name)
    ax.set_title(f'Comparison: {signal_name} across trials')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig
