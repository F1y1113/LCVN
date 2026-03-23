"""
DFoT API Wrapper for Lcvn-AC Integration
Handles communication with DFoT API and data format conversion
"""

import logging
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import pure_infer for direct DFoT inference (no HTTP fallback)
from .pure_infer import LDiTPureInference


class HistoryFrameManager:
    """
    Manages 4-frame history for DFoT API integration
    Handles sliding window updates and initialization strategies
    Supports batch processing to avoid global sharing issues
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device if device is not None else torch.device('cpu')
        self.history_latents = None  # (B, 4, latent_dim, height, width)
        self.history_actions = None  # (B, 4, action_dim)
        self.is_initialized = False
        # These will be set when first frame is processed
        self.batch_size = None
        self.latent_dim = None
        self.height = None
        self.width = None
        self.action_dim = None
        
    def initialize_history(self, initial_latent: Tensor, batch_size: int = None):
        """
        Initialize history with the first frame repeated 4 times
        Args:
            initial_latent: (B, latent_dim, height, width) - first frame latent
            batch_size: batch size (optional, will be inferred from initial_latent if not provided)
        """
        # Infer dimensions from the input tensor
        if batch_size is None:
            batch_size = initial_latent.shape[0]
        
        self.batch_size = batch_size
        self.latent_dim = initial_latent.shape[1]
        self.height = initial_latent.shape[2]
        self.width = initial_latent.shape[3]
        self.action_dim = 4  # DFoT uses 4-dim actions (dx, dy, dyaw, dt)
        
        # Repeat the initial latent 4 times for history
        self.history_latents = initial_latent.unsqueeze(1).repeat(1, 4, 1, 1, 1)  # (B, 4, latent_dim, height, width)
        
        # Initialize with stop actions (0, 0, 0, 0.1) for first 3 frames
        # Use the same device as initial_latent to ensure consistency
        target_device = initial_latent.device
        self.history_actions = torch.zeros((batch_size, 4, self.action_dim), device=target_device, dtype=torch.float32)
        self.history_actions[:, :3, 3] = 0.1  # dt = 0.1 for stop actions
        
        self.is_initialized = True
        logger.info(f"History initialized with shape: latents {self.history_latents.shape}, actions {self.history_actions.shape}")
        
    def update_history(self, new_latent: Tensor, new_action: Tensor):
        """
        Update history with sliding window
        Args:
            new_latent: (B, 4, 32, 32) - new latent frame
            new_action: (B, 4) - new action in DFoT format
        """
        if not self.is_initialized:
            raise RuntimeError("History not initialized. Call initialize_history first.")
            
        # Ensure all tensors are on the same device
        target_device = self.history_latents.device
        new_latent = new_latent.to(target_device)
        new_action = new_action.to(target_device)
            
        # Slide the history window
        self.history_latents = torch.cat([
            self.history_latents[:, 1:],  # Remove first frame
            new_latent.unsqueeze(1)       # Add new frame
        ], dim=1)
        
        self.history_actions = torch.cat([
            self.history_actions[:, 1:],  # Remove first action
            new_action.unsqueeze(1)       # Add new action
        ], dim=1)
        
    def add_frame(self, latent_frame: Tensor, action: Tensor):
        """
        Add a new frame to the history (for testing)
        Args:
            latent_frame: (B, latent_dim, height, width) - new latent frame
            action: (B, action_dim) - new action
        """
        if not self.is_initialized:
            # Initialize history if not done yet
            self.history_latents = torch.zeros((self.batch_size, 4, self.latent_dim, self.height, self.width), 
                                             device=self.device, dtype=torch.float32)
            self.history_actions = torch.zeros((self.batch_size, 4, self.action_dim), 
                                             device=self.device, dtype=torch.float32)
            self.is_initialized = True
        
        # Slide the history window
        self.history_latents = torch.cat([
            self.history_latents[:, 1:],  # Remove first frame
            latent_frame.unsqueeze(1)     # Add new frame
        ], dim=1)
        
        self.history_actions = torch.cat([
            self.history_actions[:, 1:],  # Remove first action
            action.unsqueeze(1)           # Add new action
        ], dim=1)

    def get_history(self) -> Tuple[Tensor, Tensor]:
        """
        Get current history for DFoT API call
        Returns:
            history_latents: (B, 4, latent_dim, height, width)
            history_actions: (B, 4, action_dim)
        """
        if not self.is_initialized:
            raise RuntimeError("History not initialized.")
        
        return self.history_latents, self.history_actions
        
    def reset(self):
        """Reset history manager"""
        self.history_latents = None
        self.history_actions = None
        self.is_initialized = False


class DFoTAPIWrapper:
    """
    Wrapper for DFoT API communication
    Handles data format conversion and API calls
    """
    
    def __init__(self,
                 device: Optional[str] = None,
                 dfot_checkpoint_path: Optional[str] = None,
                 vae_checkpoint_path: Optional[str] = None,
                 project_path: str = "weights/dfot",
                 vae_project_root: str = "weights/vae"):
        """
        Initialize DFoT API wrapper (pure inference mode)
        """
        self.history_manager = None
        self.dfot_inferencer = None
        # Auto-select device if not provided
        self.device_str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize DFoT pure inferencer
        try:
            original_cwd = os.getcwd()
            os.chdir(project_path)
            try:
                # Prefer function args, then environment variables, then default path
                env_dfot_ckpt = os.environ.get("DFOT_CKPT") or os.environ.get("DFOT_WEIGHTS_PT")
                env_vae_ckpt = os.environ.get("DFOT_VAE_CKPT")
                dfot_ckpt = dfot_checkpoint_path or env_dfot_ckpt or f"{project_path}/dfot_last_2025-11-19.ckpt"
                vae_ckpt = vae_checkpoint_path or env_vae_ckpt or f"{vae_project_root}/vae_epoch3_step50000.ckpt"
                self.dfot_inferencer = LDiTPureInference(
                    dfot_checkpoint_path=dfot_ckpt,
                    vae_checkpoint_path=vae_ckpt,
                    device=self.device_str
                )
                logger.info("DFoT pure inference initialized successfully")
            finally:
                os.chdir(original_cwd)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DFoT pure inference: {e}")
        
    def initialize_history_manager(self, device: torch.device):
        """Initialize the history frame manager"""
        self.history_manager = HistoryFrameManager(device)
        
        # Removed latent resizing utilities; DFoT integration now assumes 32x32 throughout
        
    def convert_lumos_to_dfot_action(self, lumos_action: Tensor) -> Tensor:
        """
        Convert LUMOS 6-dim action to DFoT 4-dim action
        Args:
            lumos_action: (B, 6) - LUMOS action format
        Returns:
            dfot_action: (B, 4) - DFoT action format (dx, dy, dyaw, dt)
        """
        batch_size = lumos_action.shape[0]
        dfot_action = torch.zeros((batch_size, 4), device=lumos_action.device, dtype=torch.float32)
        
        # Extract dx, dy, dyaw from LUMOS action
        dfot_action[:, 0] = lumos_action[:, 0]  # dx
        dfot_action[:, 1] = lumos_action[:, 1]  # dy  
        dfot_action[:, 2] = lumos_action[:, 2]  # dyaw
        dfot_action[:, 3] = 0.1                 # dt = 0.1 (fixed)
        
        return dfot_action
    
    def predict(self, latent_frames: Tensor, actions: Tensor) -> Tensor:
        """
        Predict next latent using DFoT pure inference.
        Args:
            latent_frames: (B, 4, latent_dim, height, width)
            actions: (B, 4, 4)
        Returns:
            predicted_latent: (B, latent_dim, height, width)
        """
        if self.dfot_inferencer is None:
            raise RuntimeError("DFoT inferencer not initialized.")
        # All latents are expected to be 32x32
        assert latent_frames.shape[-2:] == (32, 32), f"DFoT expects 32x32 latents, got {latent_frames.shape}"
        latents_32 = latent_frames
        # Pure inference
        predicted_latent_32 = self.dfot_inferencer.predict(
            history_latents=latents_32,
            actions=actions
        )
        # Keep 32x32 output
        return predicted_latent_32
        
    def predict_next_state(self, 
                          current_latent: Tensor, 
                          lumos_action: Tensor,
                          raw_instructions: Optional[List[str]] = None) -> Tensor:
        """
        Predict next state using DFoT API with batch processing support
        Args:
            current_latent: (B, 4, 32, 32) - current VAE latent
            lumos_action: (B, 6) - LUMOS action format
            raw_instructions: Optional language instructions
        Returns:
            next_latent: (B, 4, 32, 32) - predicted next VAE latent
        """
        batch_size = current_latent.shape[0]
        
        # Convert action format
        dfot_action = self.convert_lumos_to_dfot_action(lumos_action)
        
        # Initialize history if needed
        if not self.history_manager.is_initialized:
            self.history_manager.initialize_history(current_latent, batch_size)
            # For the first prediction, use the current action as the 4th action
            self.history_manager.history_actions[:, 3] = dfot_action
        else:
            # Update history with new frame and action
            self.history_manager.update_history(current_latent, dfot_action)
            
        # Get history for API call
        history_latents, history_actions = self.history_manager.get_history()
        
        # Call pure inference
        next_latent = self._call_dfot_api_batch(history_latents, history_actions, raw_instructions)
        logger.debug(f"DFoT prediction successful. Input shape: {current_latent.shape}, Output shape: {next_latent.shape}")
        return next_latent
            
    def _call_dfot_api_batch(self, 
                            history_latents: Tensor, 
                            history_actions: Tensor,
                            raw_instructions: Optional[List[str]] = None) -> Tensor:
        """
        Make batch API call to DFoT service
        Args:
            history_latents: (B, 4, 4, 32, 32) - LUMOS format
            history_actions: (B, 4, 4) - DFoT format
            raw_instructions: Optional language instructions
        Returns:
            next_latent: (B, 4, 32, 32) - LUMOS format
        """
        assert history_latents.shape[-2:] == (32, 32), f"DFoT expects 32x32 history latents, got {history_latents.shape}"
        history_latents_32 = history_latents
        predicted_latent_32 = self.dfot_inferencer.predict(
            history_latents=history_latents_32,
            actions=history_actions
        )
        logger.debug(f"DFoT pure inference successful. Output shape: {predicted_latent_32.shape}")
        return predicted_latent_32


        
    def reset_history(self):
        """Reset the history manager"""
        if self.history_manager:
            self.history_manager.reset()
