"""Soft Actor-Critic (SAC) Agent Implementation.

Coordinates maximum entropy policy optimization, Twin Critic updates,
temperature auto-tuning, and Polyak target smoothing.
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .config import SACConfig
from .model import GaussianPolicy, TwinQNetwork


class SACAgent:
    """Production Soft Actor-Critic agent for continuous torque control."""

    def __init__(self, config: SACConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Instantiate Actor Policy Network
        self.actor = GaussianPolicy(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            action_high=config.action_high,
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)

        # 2. Instantiate Twin Critic Networks
        self.critic = TwinQNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        # 3. Instantiate Frozen Target Critic Networks
        self.critic_target = TwinQNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # 4. Entropy Temperature Setup
        self.gamma = config.gamma
        self.tau = config.tau
        self.auto_tune_alpha = config.auto_tune_alpha

        if self.auto_tune_alpha:
            # Target entropy: H_target = -dim(A)
            self.target_entropy = -float(config.action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr_alpha)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = config.alpha

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select continuous action given single state observation."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                _, _, mean_action = self.actor.sample(state_tensor)
                return mean_action.cpu().numpy()[0]
            else:
                action, _, _ = self.actor.sample(state_tensor)
                return action.cpu().numpy()[0]

    def update_parameters(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one gradient update step across Critic, Actor, and Entropy Temperature."""
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # -----------------------------
        # 1. Update Critic Networks
        # -----------------------------
        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            min_target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            expected_q = rewards + (1.0 - dones) * self.gamma * min_target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, expected_q) + F.mse_loss(current_q2, expected_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -----------------------------
        # 2. Update Actor Network
        # -----------------------------
        sampled_actions, log_prob, _ = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)

        # Maximize: E[Q(s, a) - alpha * log_pi(a|s)] => Minimize: E[alpha * log_pi(a|s) - Q(s, a)]
        actor_loss = (self.alpha * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -----------------------------
        # 3. Auto-Tune Temperature (Alpha)
        # -----------------------------
        alpha_loss_val = 0.0
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
            alpha_loss_val = alpha_loss.item()

        # -----------------------------
        # 4. Soft Target Update (Polyak)
        # -----------------------------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss_val,
        }

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "alpha": self.alpha,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.alpha = checkpoint["alpha"]
