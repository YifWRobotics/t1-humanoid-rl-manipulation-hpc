"""Custom PPO with Lipschitz penalty regularization."""

import torch
from rsl_rl.algorithms.ppo import PPO


class LipPPO(PPO):
    """PPO with Lipschitz penalty regularization.

    This class adds a Lipschitz penalty to the policy loss to encourage
    smoother policy functions with bounded gradients.

    Features:
    - Lipschitz penalty based on gradient magnitude
    - Conditional application of penalty
    - Logging of penalty loss
    """

    def __init__(self, policy, **kwargs):
        """Initialize LipPPO.

        Args:
            policy: The policy network (ActorCritic).
            **kwargs: All standard PPO arguments plus:
                lipschitz_penalty_coeff: Coefficient for Lipschitz penalty (default: 0.1)
        """
        # Extract Lipschitz coefficient before calling parent init
        self.lipschitz_penalty_coeff = kwargs.pop('lipschitz_penalty_coeff', 0.1)
        super().__init__(policy, **kwargs)

    def _calc_lipschitz_penalty(self, obs_batch):
        """Calculate Lipschitz penalty based on policy gradient magnitude.

        The penalty is computed as the sum of squared gradients of the policy
        output with respect to the observation input. This encourages the policy
        to have bounded gradients (Lipschitz constraint).

        Args:
            obs_batch: Batch of observations with requires_grad=True

        Returns:
            Scalar Lipschitz penalty loss
        """
        batch_size, action_dim = self.actor.mean_action(obs_batch).shape

        # Compute Jacobian using autograd
        jacobian = torch.autograd.functional.jacobian(
            lambda x: torch.sum(self.actor.mean_action(x), dim=0),
            obs_batch,
            create_graph=True
        )

        # Reshape jacobian to (batch_size, action_dim, obs_dim)
        obs_dim = obs_batch.shape[1]
        jacobian = jacobian.view(action_dim, batch_size, obs_dim).permute(1, 0, 2)

        # Compute Lipschitz penalty as sum of squared gradients
        lipschitz_penalty = torch.sum(torch.square(jacobian), dim=(1, 2)).mean()

        return lipschitz_penalty

    def compute_returns(self, last_critic_obs):
        """Compute returns with custom logic (optional override).

        Args:
            last_critic_obs: Critic observations for the last step

        Returns:
            None (updates storage in-place)
        """
        # Use parent implementation
        return super().compute_returns(last_critic_obs)

    def _train_step(self):
        """Training step with PPO loss + Lipschitz penalty regularization.

        The policy loss combines:
        1. Standard PPO clipped surrogate loss
        2. Entropy regularization
        3. Lipschitz penalty (optional) to encourage smooth policies
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_lipschitz_penalty = 0

        # Iterate through learning epochs
        for epoch in range(self.num_learn_epochs):
            # Iterate through mini-batches
            for (
                actor_obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
            ) in self.storage.mini_batch_generator(self.num_mini_batches):

                # ===== PPO Surrogate Loss =====
                actions_log_prob_batch, entropy_batch = self.actor.evaluate(
                    actor_obs_batch, actions_batch
                )

                ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
                surrogate = -advantages_batch * ratio
                surrogate_clipped = -advantages_batch * torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # ===== Policy Loss with Entropy Regularization =====
                policy_loss = surrogate_loss - self.entropy_coef * entropy_batch.mean()

                # ===== Lipschitz Penalty (Optional) =====
                lipschitz_penalty = 0.0
                if self.lipschitz_penalty_coeff > 0:
                    # Enable gradient computation for observations
                    actor_obs_batch_grad = actor_obs_batch.clone().detach().requires_grad_(True)
                    lipschitz_penalty = self._calc_lipschitz_penalty(actor_obs_batch_grad)
                    policy_loss = policy_loss + self.lipschitz_penalty_coeff * lipschitz_penalty

                # ===== Actor Gradient Step =====
                self.optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # ===== Critic Loss =====
                value_loss = self.compute_critic_loss(
                    critic_obs_batch,
                    target_values_batch,
                    returns_batch
                )

                # ===== Critic Gradient Step =====
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.max_grad_norm
                )
                self.critic_optimizer.step()

                # Record losses
                mean_surrogate_loss += surrogate_loss.item()
                mean_value_loss += value_loss.item()
                if self.lipschitz_penalty_coeff > 0:
                    mean_lipschitz_penalty += lipschitz_penalty.item()

        # Average over all updates
        num_updates = self.num_learn_epochs * self.num_mini_batches
        mean_surrogate_loss /= num_updates
        mean_value_loss /= num_updates
        if self.lipschitz_penalty_coeff > 0:
            mean_lipschitz_penalty /= num_updates

        return {
            "mean_value_loss": mean_value_loss,
            "mean_surrogate_loss": mean_surrogate_loss,
            "mean_lipschitz_penalty": mean_lipschitz_penalty,
        }

    # Example: Override specific loss computation
    def compute_critic_loss(self, critic_obs_batch, target_values_batch, returns_batch):
        """Compute value function loss with optional modifications.

        Args:
            critic_obs_batch: Critic observations
            target_values_batch: Target value estimates
            returns_batch: Bootstrapped returns

        Returns:
            Value loss scalar
        """
        value_batch = self.critic(critic_obs_batch)

        # Standard clipped value loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_loss = torch.max(
                (value_batch - returns_batch) ** 2,
                (value_clipped - returns_batch) ** 2
            ).mean()
        else:
            value_loss = ((value_batch - returns_batch) ** 2).mean()

        return value_loss
