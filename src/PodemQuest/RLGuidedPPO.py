from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")


@dataclass
class DecisionStep:
    mode: str
    state_embedding: Optional[torch.Tensor]
    candidate_embeddings: list[torch.Tensor]
    candidate_count: int
    action: int
    logprob: torch.Tensor
    state_value: torch.Tensor
    reward: float = 0.0
    is_terminal: bool = False


class RolloutBuffer:
    def __init__(self):
        self.steps = []

    def clear(self):
        self.steps.clear()


class RLActorCritic(nn.Module):
    def __init__(self, gate_embedding_dim, max_backtrace_candidates, hidden_dim=64):
        super().__init__()
        self.max_backtrace_candidates = max_backtrace_candidates
        self.gate_encoder = nn.Sequential(
            nn.Linear(gate_embedding_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mode_embedding = nn.Embedding(2, hidden_dim)
        self.backtrace_actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, max_backtrace_candidates),
        )
        self.propagation_actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_gate(self, gate_embedding, mode):
        gate_tensor = gate_embedding.to(device=device, dtype=torch.float32).unsqueeze(0)
        mode_tensor = torch.tensor([mode], dtype=torch.long, device=device)
        return (
            self.gate_encoder(gate_tensor)
            + self.mode_embedding(mode_tensor)
        ).squeeze(0)

    def backtrace_logits(self, gate_embedding):
        state_repr = self.encode_gate(gate_embedding, mode=0)
        logits = self.backtrace_actor(state_repr)
        state_value = self.critic(state_repr).squeeze(-1)
        return logits, state_value

    def propagation_logits(self, candidate_gate_embeddings):
        candidate_reprs = []
        for gate_embedding in candidate_gate_embeddings:
            candidate_reprs.append(self.encode_gate(gate_embedding, mode=1))
        candidate_reprs = torch.stack(candidate_reprs, dim=0)
        logits = self.propagation_actor(candidate_reprs).squeeze(-1)
        state_repr = candidate_reprs.mean(dim=0)
        state_value = self.critic(state_repr).squeeze(-1)
        return logits, state_value

    def evaluate_step(self, step):
        if step.mode == "backtrace":
            logits, state_value = self.backtrace_logits(step.state_embedding)
            logits = logits[: step.candidate_count]
        else:
            logits, state_value = self.propagation_logits(step.candidate_embeddings)

        dist = Categorical(logits=logits)
        action_tensor = torch.tensor(step.action, dtype=torch.long, device=device)
        logprob = dist.log_prob(action_tensor)
        entropy = dist.entropy()
        return logprob, state_value, entropy


class RLGuidedPPOAgent:
    def __init__(
        self,
        gate_embedding_dim,
        max_backtrace_candidates,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        k_epochs=8,
        eps_clip=0.2,
    ):
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.buffer = RolloutBuffer()
        self.last_selected_step_idx = None
        self.last_selected_mode = None

        self.policy = RLActorCritic(
            gate_embedding_dim=gate_embedding_dim,
            max_backtrace_candidates=max_backtrace_candidates,
        ).to(device)
        self.policy_old = RLActorCritic(
            gate_embedding_dim=gate_embedding_dim,
            max_backtrace_candidates=max_backtrace_candidates,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.gate_encoder.parameters(), "lr": lr_actor},
                {"params": self.policy.mode_embedding.parameters(), "lr": lr_actor},
                {"params": self.policy.backtrace_actor.parameters(), "lr": lr_actor},
                {"params": self.policy.propagation_actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )
        self.mse_loss = nn.MSELoss()

    def _gate_embedding(self, gate):
        if gate.deepgate_embedding is None:
            raise ValueError(f"Gate '{gate.outputpin}' is missing a fixed DeepGate embedding.")
        return gate.deepgate_embedding

    def select_backtrace_action(self, objective_gate, candidate_gates):
        objective_embedding = self._gate_embedding(objective_gate)
        logits, state_value = self.policy_old.backtrace_logits(objective_embedding)
        logits = logits[: len(candidate_gates)]
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        self.buffer.steps.append(
            DecisionStep(
                mode="backtrace",
                state_embedding=objective_embedding.detach().cpu(),
                candidate_embeddings=[],
                candidate_count=len(candidate_gates),
                action=int(action.item()),
                logprob=logprob.detach(),
                state_value=state_value.detach(),
            )
        )
        self.last_selected_step_idx = len(self.buffer.steps) - 1
        self.last_selected_mode = "backtrace"
        return candidate_gates[int(action.item())]

    def select_propagation_action(self, frontier_gates):
        candidate_embeddings = [self._gate_embedding(gate) for gate in frontier_gates]
        logits, state_value = self.policy_old.propagation_logits(candidate_embeddings)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        self.buffer.steps.append(
            DecisionStep(
                mode="propagation",
                state_embedding=None,
                candidate_embeddings=[embedding.detach().cpu() for embedding in candidate_embeddings],
                candidate_count=len(frontier_gates),
                action=int(action.item()),
                logprob=logprob.detach(),
                state_value=state_value.detach(),
            )
        )
        self.last_selected_step_idx = len(self.buffer.steps) - 1
        self.last_selected_mode = "propagation"
        return frontier_gates[int(action.item())]

    def add_reward(self, reward):
        if self.buffer.steps:
            self.buffer.steps[-1].reward += reward

    def add_reward_to_step(self, step_idx, reward):
        if step_idx is None:
            return
        if 0 <= step_idx < len(self.buffer.steps):
            self.buffer.steps[step_idx].reward += reward

    def finish_episode(self, final_reward):
        if not self.buffer.steps:
            return
        self.buffer.steps[-1].reward += final_reward
        self.buffer.steps[-1].is_terminal = True

    def update(self):
        if not self.buffer.steps:
            return

        returns = []
        discounted_reward = 0.0
        for step in reversed(self.buffer.steps):
            if step.is_terminal:
                discounted_reward = 0.0
            discounted_reward = step.reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        old_logprobs = torch.stack([step.logprob for step in self.buffer.steps]).to(device)
        old_state_values = (
            torch.stack([step.state_value for step in self.buffer.steps]).to(device).squeeze(-1)
        )
        advantages = returns.detach() - old_state_values.detach()

        for _ in range(self.k_epochs):
            losses = []
            for idx, step in enumerate(self.buffer.steps):
                logprob, state_value, entropy = self.policy.evaluate_step(step)
                ratio = torch.exp(logprob - old_logprobs[idx].detach())
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[idx]
                value_loss = self.mse_loss(state_value.squeeze(), returns[idx])
                loss = -torch.min(surr1, surr2) + 0.5 * value_loss - 0.01 * entropy
                losses.append(loss)

            total_loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.policy_old.load_state_dict(state_dict)
        self.policy.load_state_dict(state_dict)
