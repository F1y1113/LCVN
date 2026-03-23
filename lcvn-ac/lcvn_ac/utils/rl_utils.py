import torch


def lambda_return(rewards, values, lambda_=0.95, gamma=0.99, inverse=False):
    """
    \lambda return recursive definition, according to Dreamerv2 paper:

    if t<H:
        V^\lambda_t = r_t + \gamma * ((1-\lambda)*v_hat_t+1 + \lambda * V^\lambda_t+1)
    elif t=H:
        V^\lambda_t = r_t + \gamma * v_hat[-1]

    according to Director paper,
    if t=H: V^\lambda_t = v_hat[-1]
    """
    # Ensure consistent tensor dimensionality to avoid stack errors
    # Start from the bootstrap value at the last timestep
    R = torch.atleast_1d(values[-1])
    returns = [R]
    # ignores last reward and first value
    rewards_less_last = rewards[:-1]
    values_less_first = values[1:]
    for r_t, v_tplus1 in zip(rewards_less_last[::-1], values_less_first[::-1]):
        # Make sure participating tensors are at least 1D to keep shapes consistent
        r_t = torch.atleast_1d(r_t)
        v_tplus1 = torch.atleast_1d(v_tplus1)
        R = r_t + gamma * ((1 - lambda_) * v_tplus1 + lambda_ * R)
        R = torch.atleast_1d(R)
        returns.insert(0, R)
    returns = torch.stack(returns)
    return returns



def MC_return(latent_rewards, bootstrap, norm=False, gamma=0.99, eps=1e-8):
    latent_rewards[-1] += bootstrap
    R = torch.zeros((len(latent_rewards[0])), device=latent_rewards[0].device)
    returns = []
    for r in latent_rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.stack(returns)
    if norm:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns


def advantage(value_buffer: torch.Tensor, returns: torch.Tensor, norm: bool) -> torch.Tensor:
    # values.shape == returns.shape == (T,B)
    values = torch.stack(value_buffer)
    advantage = returns - values
    if norm:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    return advantage


def max_cos(input, target):
    n_i = torch.norm(input, dim=-1, keepdim=True)
    n_t = torch.norm(target, dim=-1, keepdim=True)
    norms = torch.cat((n_i, n_t), dim=-1)
    max_norm = torch.max(norms, dim=-1)[0]
    dot_prod = (input * target).sum(dim=-1)
    max_cos = dot_prod / torch.square(max_norm)

    return max_cos


def action_mse(actions, ac_actions):
    # T, B, action_dim
    mse_per_step = []
    for i in range(actions.size(0)):
        squared_errors = (actions[i] - ac_actions[i]) ** 2
        mse = squared_errors.mean()
        mse_per_step.append(mse)
    return mse_per_step
