import numpy as np
import h5py
import yaml
import time
import shutil
from pathlib import Path
from collections import defaultdict

class DataLogger:
    def __init__(self, trial_name, output_dir='data/', buffer_size=10000):
        self.trial_name = trial_name
        self.output_dir = Path(output_dir)
        self.buffer_size = buffer_size
        
        self.trial_dir = self.output_dir / trial_name
        
        # Delete old trial directory if it exists
        if self.trial_dir.exists():
            shutil.rmtree(self.trial_dir)
        
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        
        self._buffers = defaultdict(lambda: {'timestamps': [], 'data': defaultdict(list)})
        self._schemas = {}
        self._start_time = time.time()
        
        self.metadata = {
            'trial_name': trial_name,
            'start_time': self._start_time,
            'end_time': None
        }
    
    def log_bundle(self, bundle_name, data_dict):
        timestamp = time.time()
        
        if bundle_name not in self._schemas:
            self._schemas[bundle_name] = {key: np.array(val).shape 
                                          for key, val in data_dict.items()}
        else:
            for key, val in data_dict.items():
                expected_shape = self._schemas[bundle_name].get(key)
                actual_shape = np.array(val).shape
                if expected_shape != actual_shape:
                    raise ValueError(f"Shape mismatch in bundle '{bundle_name}', signal '{key}': "
                                   f"expected {expected_shape}, got {actual_shape}")
        
        # Add timestamp first
        self._buffers[bundle_name]['timestamps'].append(timestamp)
        
        # Then add all data
        for key, val in data_dict.items():
            self._buffers[bundle_name]['data'][key].append(np.array(val))
        
        # Check buffer size AFTER adding all data to ensure consistency
        if len(self._buffers[bundle_name]['timestamps']) >= self.buffer_size:
            self._flush_bundle(bundle_name)
    
    def _flush_bundle(self, bundle_name):
        buffer = self._buffers[bundle_name]
        
        if len(buffer['timestamps']) == 0:
            return
        
        # Verify all data arrays have the same length as timestamps
        n_samples = len(buffer['timestamps'])
        for key, values in buffer['data'].items():
            if len(values) != n_samples:
                raise ValueError(f"Data length mismatch in bundle '{bundle_name}': "
                               f"timestamps has {n_samples} samples but '{key}' has {len(values)}")
        
        filepath = self.trial_dir / f"{bundle_name}.h5"
        mode = 'a' if filepath.exists() else 'w'
        
        with h5py.File(filepath, mode) as f:
            timestamps = np.array(buffer['timestamps'])
            
            if 'timestamps' in f:
                existing = f['timestamps'][:]
                combined = np.concatenate([existing, timestamps])
                del f['timestamps']
                f.create_dataset('timestamps', data=combined)
            else:
                f.create_dataset('timestamps', data=timestamps)
            
            for key, values in buffer['data'].items():
                arr = np.array(values)
                
                if key in f:
                    existing = f[key][:]
                    combined = np.concatenate([existing, arr], axis=0)
                    del f[key]
                    f.create_dataset(key, data=combined)
                else:
                    f.create_dataset(key, data=arr)
        
        # Clear buffers
        buffer['timestamps'].clear()
        buffer['data'].clear()
    
    def save(self):
        # Flush all remaining data
        for bundle_name in list(self._buffers.keys()):
            self._flush_bundle(bundle_name)
        
        self.metadata['end_time'] = time.time()
        self.metadata['duration'] = self.metadata['end_time'] - self.metadata['start_time']
        self.metadata['schemas'] = {name: {k: list(v) for k, v in schema.items()} 
                                    for name, schema in self._schemas.items()}
        
        with open(self.trial_dir / 'metadata.yaml', 'w') as f:
            yaml.dump(self.metadata, f)
    
    def clear(self):
        self._buffers.clear()
        self._schemas.clear()
    
    def get_summary(self):
        summary = {
            'trial_name': self.trial_name,
            'duration': self.metadata.get('duration', 0),
            'bundles': {}
        }
        
        for bundle_name in self._schemas.keys():
            filepath = self.trial_dir / f"{bundle_name}.h5"
            if filepath.exists():
                with h5py.File(filepath, 'r') as f:
                    n_samples = len(f['timestamps'])
                    signals = list(f.keys())
                    signals.remove('timestamps')
                    summary['bundles'][bundle_name] = {
                        'n_samples': n_samples,
                        'signals': signals
                    }
        
        return summary
    
    @staticmethod
    def load(trial_path, bundle_name):
        trial_path = Path(trial_path)
        filepath = trial_path / f"{bundle_name}.h5"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Bundle '{bundle_name}' not found in {trial_path}")
        
        data = {}
        with h5py.File(filepath, 'r') as f:
            data['timestamps'] = f['timestamps'][:]
            for key in f.keys():
                if key != 'timestamps':
                    data[key] = f[key][:]
        
        # Verify all data has same length as timestamps
        n_samples = len(data['timestamps'])
        for key, values in data.items():
            if key != 'timestamps' and len(values) != n_samples:
                raise ValueError(f"Data corruption: '{key}' has {len(values)} samples "
                               f"but timestamps has {n_samples}")
        
        return data
    
    @staticmethod
    def load_all(trial_path):
        """
        Load all bundles from a trial directory.
        """
        trial_path = Path(trial_path)
        
        if not trial_path.exists():
            raise FileNotFoundError(f"Trial directory not found: {trial_path}")
        
        # Find all .h5 files in the trial directory
        bundle_files = list(trial_path.glob("*.h5"))
        
        if len(bundle_files) == 0:
            raise FileNotFoundError(f"No bundle files found in {trial_path}")
        
        all_bundles = {}
        
        for filepath in bundle_files:
            bundle_name = filepath.stem  # Get filename without extension
            
            bundle_data = {}
            with h5py.File(filepath, 'r') as f:
                bundle_data['timestamps'] = f['timestamps'][:]
                for key in f.keys():
                    if key != 'timestamps':
                        bundle_data[key] = f[key][:]
            
            # Verify data integrity
            n_samples = len(bundle_data['timestamps'])
            for key, values in bundle_data.items():
                if key != 'timestamps' and len(values) != n_samples:
                    raise ValueError(f"Data corruption in bundle '{bundle_name}': "
                                f"'{key}' has {len(values)} samples "
                                f"but timestamps has {n_samples}")
            
            all_bundles[bundle_name] = bundle_data
        
        return all_bundles