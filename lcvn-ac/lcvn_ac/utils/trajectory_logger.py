"""
Trajectory Logger for LUMOS Training Data Flow Tracking

This module provides comprehensive logging functionality to track the complete data flow
of trajectories during LUMOS training, from raw data loading to final action prediction.
"""

import os
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import threading


class TrajectoryLogger:
    """
    Singleton logger for tracking trajectory data flow in LUMOS training.
    
    Features:
    - Thread-safe logging
    - Configurable enable/disable
    - Detailed data shape and value logging
    - Automatic trajectory counting
    - JSON and text output formats
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TrajectoryLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        # Default: logging disabled; enable only via explicit call or env var
        # Global default off; enable only when env variable is set
        self.enabled = str(os.environ.get("LCVN_AC_TRAJECTORY_LOGGING", "0")).lower() in ("1", "true", "yes")
        self.log_dir = Path("trajectory_logs")
        
        # Trajectory tracking
        self.trajectory_count = 0
        self.current_trajectory_id = None
        self.max_trajectories_to_log = 3  # Log first 3 trajectories by default
        # Size guard to avoid oversized log files
        self.max_log_file_size_mb = 512  # stop appending when txt exceeds this size
        self._size_guard_triggered = False
        
        # Log files
        # Lazy init: create file paths only when enabled
        self.log_file = None
        self.json_file = None
        
        # Data storage for current trajectory
        self.current_trajectory_data = {}
        
        # Thread safety
        self._file_lock = threading.Lock()

        # If enabled via environment variable, set up files and header now
        if self.enabled:
            self._setup_files_and_header()

    def _setup_files_and_header(self):
        """Create log directory and files, then write header (only when enabled)."""
        try:
            self.log_dir.mkdir(exist_ok=True)
        except Exception:
            # On directory creation exception, remain disabled to avoid interrupting training
            self.enabled = False
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"trajectory_log_{timestamp}.txt"
        self.json_file = self.log_dir / f"trajectory_data_{timestamp}.json"
        self._write_header()
    
    def _write_header(self):
        """Write log file header with configuration info."""
        header = f"""
{'='*80}
Lcvn-AC Trajectory Data Flow Logger
Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Log Directory: {self.log_dir}
Max Trajectories to Log: {self.max_trajectories_to_log}
{'='*80}

"""
        with self._file_lock:
            with open(self.log_file, 'w') as f:
                f.write(header)
    
    def enable(self):
        """Enable trajectory logging."""
        if self.enabled:
            return
        self.enabled = True
        # When enabling, create directory and log files and write header
        self._setup_files_and_header()
        # Record enable event
        self.log_event("SYSTEM", "Trajectory logging ENABLED")
    
    def disable(self):
        """Disable trajectory logging."""
        # Force disable, do not write any log
        self.enabled = False
    
    def should_log_trajectory(self) -> bool:
        """Check if current trajectory should be logged."""
        return (self.enabled and 
                self.trajectory_count < self.max_trajectories_to_log)
    
    def start_new_trajectory(self, trajectory_info: Dict[str, Any] = None):
        """Start logging a new trajectory."""
        if not self.enabled:
            return
            
        self.trajectory_count += 1
        self.current_trajectory_id = f"trajectory_{self.trajectory_count:04d}"
        
        if self.should_log_trajectory():
            self.current_trajectory_data = {
                'trajectory_id': self.current_trajectory_id,
                'trajectory_number': self.trajectory_count,
                'start_time': datetime.now().isoformat(),
                'info': trajectory_info or {},
                'data_flow': []
            }
            
            self.log_event("TRAJECTORY_START", 
                          f"Starting trajectory {self.current_trajectory_id}",
                          trajectory_info)
    
    def log_event(self, stage: str, message: str, data: Any = None):
        """Log a general event with optional data."""
        if not self.enabled:
            return
            
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        log_entry = f"[{timestamp}] [{stage}] {message}"
        
        if data is not None:
            if isinstance(data, (torch.Tensor, np.ndarray)):
                log_entry += f"\n  Shape: {data.shape}, Dtype: {data.dtype}"
                if hasattr(data, 'device'):
                    log_entry += f", Device: {data.device}"
                # Log some sample values for small arrays/tensors
                num_elements = data.numel() if isinstance(data, torch.Tensor) else data.size
                if num_elements <= 20:
                    log_entry += f"\n  Values: {data.flatten()}"
                else:
                    log_entry += f"\n  Sample values: {data.flatten()[:10]}..."
            elif isinstance(data, dict):
                log_entry += f"\n  Data keys: {list(data.keys())}"
                for key, value in data.items():
                    if isinstance(value, (torch.Tensor, np.ndarray)):
                        log_entry += f"\n    {key}: shape={value.shape}, dtype={value.dtype}"
                    else:
                        log_entry += f"\n    {key}: {type(value).__name__} = {value}"
            else:
                log_entry += f"\n  Data: {data}"
        
        log_entry += "\n" + "-" * 60 + "\n"
        
        # File size guard: stop appending TEXT when exceeding limit (JSON continues)
        try:
            current_size_mb = (self.log_file.stat().st_size if self.log_file.exists() else 0) / (1024 * 1024)
        except Exception:
            current_size_mb = 0

        if current_size_mb >= self.max_log_file_size_mb:
            if not self._size_guard_triggered:
                # Write one-time notice and disable further logging to prevent GB-sized files
                self._size_guard_triggered = True
                notice = f"[{timestamp}] [SYSTEM] Log file size {current_size_mb:.2f}MB exceeds limit {self.max_log_file_size_mb}MB. Further logging suppressed.\n" + ("-" * 60) + "\n"
                with self._file_lock:
                    try:
                        with open(self.log_file, 'a') as f:
                            f.write(notice)
                    except Exception:
                        pass
            # Suppress further TEXT logging but keep JSON structured logging active
            return

        with self._file_lock:
            with open(self.log_file, 'a') as f:
                f.write(log_entry)

        # Also append a lightweight structured event to JSON data_flow when trajectory active
        try:
            if hasattr(self, 'current_trajectory_data') and self.current_trajectory_data:
                self.current_trajectory_data['data_flow'].append({
                    'type': 'EVENT',
                    'stage': stage,
                    'timestamp': datetime.now().isoformat(),
                    'message': message
                })
        except Exception:
            pass
    
    def log_data_loading(self, stage: str, data: Dict[str, Any], 
                        additional_info: Dict[str, Any] = None):
        """Log data loading events with detailed information."""
        if not self.should_log_trajectory():
            return
            
        log_data = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'data_info': {},
            'additional_info': additional_info or {}
        }
        
        # Extract detailed information about loaded data
        for key, value in data.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                stats = {
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'min_val': None,
                    'max_val': None,
                    'mean_val': None,
                }
                if isinstance(value, torch.Tensor):
                    num_elements = value.numel()
                    is_numeric = value.is_floating_point() or value.is_complex()
                    if num_elements > 0 and is_numeric:
                        try:
                            stats['min_val'] = float(value.min().item())
                            stats['max_val'] = float(value.max().item())
                            stats['mean_val'] = float(value.mean().item())
                        except Exception:
                            # Fallback: skip stats if dtype unsupported
                            pass
                    if hasattr(value, 'device'):
                        stats['device'] = str(value.device)
                else:  # numpy.ndarray
                    num_elements = value.size
                    # Compute stats only for numeric arrays (skip bool and object)
                    try:
                        is_numeric = (np.issubdtype(value.dtype, np.floating) or np.issubdtype(value.dtype, np.integer))
                    except Exception:
                        is_numeric = False
                    if num_elements > 0 and is_numeric:
                        try:
                            stats['min_val'] = float(value.min())
                            stats['max_val'] = float(value.max())
                            stats['mean_val'] = float(value.mean())
                        except Exception:
                            pass
                log_data['data_info'][key] = stats
            else:
                log_data['data_info'][key] = {
                    'type': type(value).__name__,
                    'value': str(value)[:100]  # Truncate long strings
                }
        
        # Add to current trajectory data
        if hasattr(self, 'current_trajectory_data') and self.current_trajectory_data:
            self.current_trajectory_data['data_flow'].append(log_data)
        
        # Write to text log
        message = f"Data loading at stage: {stage}"
        self.log_event("DATA_LOADING", message, data)
    
    def log_window_processing(self, original_data: Dict[str, Any], 
                            windowed_data: Dict[str, Any],
                            window_size: int, pad_size: int = 0):
        """Log window processing and padding operations."""
        if not self.should_log_trajectory():
            return
            
        processing_info = {
            'window_size': window_size,
            'pad_size': pad_size,
            'original_shapes': {},
            'windowed_shapes': {}
        }
        
        for key in original_data.keys():
            if isinstance(original_data[key], (torch.Tensor, np.ndarray)):
                processing_info['original_shapes'][key] = list(original_data[key].shape)
        
        for key in windowed_data.keys():
            if isinstance(windowed_data[key], (torch.Tensor, np.ndarray)):
                processing_info['windowed_shapes'][key] = list(windowed_data[key].shape)
        
        message = f"Window processing: size={window_size}, padding={pad_size}"
        self.log_event("WINDOW_PROCESSING", message, processing_info)
        
        # Log detailed windowed data
        self.log_data_loading("WINDOWED_DATA", windowed_data, processing_info)
    
    def log_model_forward(self, stage: str, inputs: Dict[str, Any], 
                         outputs: Dict[str, Any] = None):
        """Log model forward pass data."""
        if not self.should_log_trajectory():
            return
            
        message = f"Model forward pass: {stage}"
        
        # Log inputs
        self.log_event(f"MODEL_INPUT_{stage}", message, inputs)
        
        # Log outputs if provided
        if outputs is not None:
            self.log_event(f"MODEL_OUTPUT_{stage}", f"Output from {stage}", outputs)

        # Append structured JSON entry with summarized inputs/outputs
        def _summarize(x: Any, include_values: bool = True, max_elements: int = 1000) -> Dict[str, Any]:
            try:
                if isinstance(x, torch.Tensor):
                    info = {
                        'shape': list(x.shape),
                        'dtype': str(x.dtype),
                        'device': str(x.device) if hasattr(x, 'device') else 'cpu'
                    }
                    numel = int(x.numel())
                    if numel > 0 and (x.is_floating_point() or x.is_complex()):
                        info.update({
                            'min_val': float(x.min().item()),
                            'max_val': float(x.max().item()),
                            'mean_val': float(x.mean().item())
                        })
                    if include_values and numel <= max_elements:
                        info['values'] = x.detach().cpu().numpy().tolist()
                    elif include_values and numel > 0:
                        # sample first N values for traceability
                        flat = x.detach().flatten().cpu().numpy()
                        n = min( min(max_elements, 100), flat.shape[0] )
                        info['sample_values'] = flat[:n].tolist()
                    return info
                elif isinstance(x, np.ndarray):
                    info = {
                        'shape': list(x.shape),
                        'dtype': str(x.dtype),
                        'device': 'cpu'
                    }
                    if x.size > 0 and np.issubdtype(x.dtype, np.number):
                        info.update({
                            'min_val': float(np.min(x)),
                            'max_val': float(np.max(x)),
                            'mean_val': float(np.mean(x))
                        })
                    if include_values and x.size <= max_elements:
                        info['values'] = x.tolist()
                    elif include_values and x.size > 0:
                        flat = x.reshape(-1)
                        n = min( min(max_elements, 100), flat.shape[0] )
                        info['sample_values'] = flat[:n].tolist()
                    return info
                elif isinstance(x, (list, tuple)):
                    arr = np.array(x)
                    return _summarize(arr, include_values, max_elements)
                elif isinstance(x, (int, float)):
                    return {
                        'shape': [],
                        'dtype': type(x).__name__,
                        'device': 'cpu',
                        'values': float(x)
                    }
                elif isinstance(x, dict):
                    # recursively summarize dict
                    return {k: _summarize(v, include_values, max_elements) for k, v in x.items()}
                else:
                    return {'type': type(x).__name__, 'value': str(x)}
            except Exception:
                return {'type': type(x).__name__, 'value': 'unavailable'}

        try:
            if hasattr(self, 'current_trajectory_data') and self.current_trajectory_data:
                entry = {
                    'type': 'MODEL_FORWARD',
                    'stage': stage,
                    'timestamp': datetime.now().isoformat(),
                    'inputs': {k: _summarize(v, include_values=True) for k, v in (inputs or {}).items()}
                }
                if outputs is not None:
                    entry['outputs'] = {k: _summarize(v, include_values=True) for k, v in outputs.items()}
                self.current_trajectory_data['data_flow'].append(entry)
        except Exception:
            pass
    
    def log_latent_output(self, latent_data: Union[torch.Tensor, np.ndarray, list, tuple, int, float], stage: str = "PURE_INFER"):
        """Log latent image outputs from pure inference.
        
        Robust to non-tensor inputs (ints, floats, lists) to prevent AttributeError
        when upstream passes non-tensor values.
        """
        if not self.should_log_trajectory():
            return

        def _summarize(x: Union[torch.Tensor, np.ndarray, list, tuple, int, float]) -> Dict[str, Any]:
            try:
                if isinstance(x, torch.Tensor):
                    # Ensure numeric stats only for floating/complex tensors
                    stats = {
                        'shape': list(x.shape),
                        'dtype': str(x.dtype),
                        'device': str(x.device) if hasattr(x, 'device') else 'cpu'
                    }
                    if x.numel() > 0 and (x.is_floating_point() or x.is_complex()):
                        stats.update({
                            'min_val': float(x.min().item()),
                            'max_val': float(x.max().item()),
                            'mean_val': float(x.mean().item()),
                            'std_val': float(x.std().item())
                        })
                    else:
                        stats.update({
                            'min_val': None,
                            'max_val': None,
                            'mean_val': None,
                            'std_val': None
                        })
                    return stats
                elif isinstance(x, np.ndarray):
                    stats = {
                        'shape': list(x.shape),
                        'dtype': str(x.dtype),
                        'device': 'cpu'
                    }
                    if x.size > 0 and np.issubdtype(x.dtype, np.number):
                        stats.update({
                            'min_val': float(np.min(x)),
                            'max_val': float(np.max(x)),
                            'mean_val': float(np.mean(x)),
                            'std_val': float(np.std(x))
                        })
                    else:
                        stats.update({
                            'min_val': None,
                            'max_val': None,
                            'mean_val': None,
                            'std_val': None
                        })
                    return stats
                elif isinstance(x, (list, tuple)):
                    arr = np.array(x)
                    return _summarize(arr)
                elif isinstance(x, (int, float)):
                    val = float(x)
                    return {
                        'shape': [],
                        'dtype': type(x).__name__,
                        'device': 'cpu',
                        'min_val': val,
                        'max_val': val,
                        'mean_val': val,
                        'std_val': 0.0
                    }
                else:
                    # Fallback: represent as string
                    return {
                        'shape': [],
                        'dtype': type(x).__name__,
                        'device': 'cpu',
                        'min_val': None,
                        'max_val': None,
                        'mean_val': None,
                        'std_val': None
                    }
            except Exception:
                return {
                    'shape': [],
                    'dtype': type(x).__name__,
                    'device': 'cpu',
                    'min_val': None,
                    'max_val': None,
                    'mean_val': None,
                    'std_val': None
                }

        latent_info = _summarize(latent_data)
        message = f"Latent output from {stage}"
        self.log_event("LATENT_OUTPUT", message, latent_info)

        # Append structured latent info to JSON data_flow
        try:
            if hasattr(self, 'current_trajectory_data') and self.current_trajectory_data:
                self.current_trajectory_data['data_flow'].append({
                    'type': 'LATENT_OUTPUT',
                    'stage': stage,
                    'timestamp': datetime.now().isoformat(),
                    'latent': latent_info
                })
        except Exception:
            pass
    
    def log_action_prediction(self, predicted_actions: Union[torch.Tensor, np.ndarray, list, tuple, int, float], 
                            ground_truth_actions: Union[torch.Tensor, np.ndarray, list, tuple, int, float] = None):
        """Log predicted actions and optionally ground truth for comparison.
        
        Robust to non-tensor inputs (e.g., Python ints/floats/lists) to avoid
        AttributeError when upstream code passes scalar labels.
        """
        if not self.should_log_trajectory():
            return

        def _format_actions(x: Union[torch.Tensor, np.ndarray, list, tuple, int, float]) -> Dict[str, Any]:
            """Format action data into a dict with shape, dtype and values."""
            try:
                if isinstance(x, torch.Tensor):
                    return {
                        'shape': list(x.shape),
                        'dtype': str(x.dtype),
                        'values': x.detach().cpu().numpy().tolist()
                    }
                elif isinstance(x, np.ndarray):
                    return {
                        'shape': list(x.shape),
                        'dtype': str(x.dtype),
                        'values': x.tolist()
                    }
                elif isinstance(x, (list, tuple)):
                    arr = np.array(x)
                    return {
                        'shape': list(arr.shape),
                        'dtype': str(arr.dtype),
                        'values': arr.tolist()
                    }
                elif isinstance(x, (int, float)):
                    return {
                        'shape': [],
                        'dtype': type(x).__name__,
                        'values': x
                    }
                else:
                    arr = np.array(x)
                    return {
                        'shape': list(arr.shape),
                        'dtype': str(arr.dtype),
                        'values': arr.tolist()
                    }
            except Exception:
                return {
                    'shape': [],
                    'dtype': type(x).__name__,
                    'values': str(x)
                }

        action_info = {
            'predicted_actions': _format_actions(predicted_actions)
        }

        if ground_truth_actions is not None:
            action_info['ground_truth_actions'] = _format_actions(ground_truth_actions)

        message = "Action prediction completed"
        self.log_event("ACTION_PREDICTION", message, action_info)

        # Append structured actions to JSON data_flow
        try:
            if hasattr(self, 'current_trajectory_data') and self.current_trajectory_data:
                self.current_trajectory_data['data_flow'].append({
                    'type': 'ACTION_PREDICTION',
                    'timestamp': datetime.now().isoformat(),
                    'actions': action_info
                })
        except Exception:
            pass
    
    def finish_trajectory(self):
        """Finish logging current trajectory and save to JSON."""
        if not self.should_log_trajectory() or not hasattr(self, 'current_trajectory_data'):
            return
            
        if self.current_trajectory_data:
            self.current_trajectory_data['end_time'] = datetime.now().isoformat()
            
            # Save to JSON file
            json_data = []
            if self.json_file.exists():
                try:
                    with open(self.json_file, 'r') as f:
                        json_data = json.load(f)
                except:
                    json_data = []
            
            json_data.append(self.current_trajectory_data)
            
            with self._file_lock:
                with open(self.json_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
            
            self.log_event("TRAJECTORY_END", 
                          f"Finished trajectory {self.current_trajectory_id}")
            
            # Clear current trajectory data
            self.current_trajectory_data = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            'enabled': self.enabled,
            'trajectory_count': self.trajectory_count,
            'max_trajectories_to_log': self.max_trajectories_to_log,
            'log_file': str(self.log_file),
            'json_file': str(self.json_file),
            'current_trajectory_id': self.current_trajectory_id
        }


# Global logger instance
trajectory_logger = TrajectoryLogger()


# Convenience functions
def enable_trajectory_logging():
    """Enable trajectory logging globally."""
    trajectory_logger.enable()


def disable_trajectory_logging():
    """Disable trajectory logging globally."""
    trajectory_logger.disable()


def log_trajectory_event(stage: str, message: str, data: Any = None):
    """Log a trajectory event."""
    trajectory_logger.log_event(stage, message, data)


def start_trajectory_logging(trajectory_info: Dict[str, Any] = None):
    """Start logging a new trajectory."""
    trajectory_logger.start_new_trajectory(trajectory_info)


def finish_trajectory_logging():
    """Finish logging current trajectory."""
    trajectory_logger.finish_trajectory()
