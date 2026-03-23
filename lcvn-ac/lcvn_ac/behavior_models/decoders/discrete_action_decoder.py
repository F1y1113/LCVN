import torch
import torch.nn as nn
from lcvn_ac.behavior_models.decoders.action_decoder import ActionDecoder

class DiscreteActionDecoder(ActionDecoder):
    def __init__(self, perceptual_features, latent_goal_features, plan_features, hidden_size, action_dim, continuous_dim=4):
        super().__init__()
        in_features = perceptual_features + latent_goal_features + plan_features

        # Shared layers
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        
        # Discrete action head
        self.fc_discrete = nn.Linear(hidden_size, action_dim)

    def forward(self, latent_current, latent_goal, latent_plan=None):
        if latent_plan is not None:
            x = torch.cat([latent_current, latent_goal, latent_plan], dim=-1)
        else:
            x = torch.cat([latent_current, latent_goal], dim=-1)

        # Shared feature extraction
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Get discrete action logits
        logits = self.fc_discrete(x)
        
        return logits

    def get_action(self, latent_current, latent_goal, latent_plan=None):
        logits = self.forward(latent_current, latent_goal, latent_plan)
        
        # Calculate discrete action and probabilities
        probs = torch.softmax(logits, dim=-1)
        discrete_action = probs.argmax(dim=-1)
        
        # For entropy calculation (if needed)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # Return everything in a dictionary format for consistency
        return {
            "discrete_logits": logits,
            "discrete_action": discrete_action,
            "entropy": entropy
        }
