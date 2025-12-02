import torch


class DiffusionProcess:

    def __init__(self, beta_schedule):
        self.beta = beta_schedule
        self.n_steps = len(beta_schedule)
        self.t = torch.linspace(0, 1, self.n_steps)

        dt = self.t[1] - self.t[0]
        self.sigma_squared = torch.cumsum(self.beta, dim=0) * dt

        total_integral = self.sigma_squared[-1]
        self.sigma_bar_squared = total_integral - self.sigma_squared

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.n_steps, (batch_size,))

    @staticmethod
    def _extract(arr, t, x, dtype=torch.float32, device=torch.device("cpu"), ndim=4):
        if x is not None:
            dtype = x.dtype
            device = x.device
            ndim = x.ndim
        out = torch.as_tensor(arr, dtype=dtype, device=device).gather(0, t)
        return out.reshape((-1,) + (1,) * (ndim - 1))

    """
    From equation (11) in the paper

    X_t = N(mu_t, sigma_t^2 I)
    mu_t = (sigma_bar_t^2 / (sigma_bar_t^2 + sigma_t^2)) * X_0 + (sigma_t^2 / (sigma_bar_t^2 + sigma_t^2)) * X_1
    Sigma_t = sigma_t^2 * sigma_bar_t^2 / (sigma_bar_t^2 + sigma_t^2)
    """

    def sample_xt(self, x0, x1, t):
        sigma_sq = self._extract(self.sigma_squared, t, x0)
        sigma_bar_sq = self._extract(self.sigma_bar_squared, t, x0)

        mu_t = (sigma_bar_sq / (sigma_bar_sq + sigma_sq)) * x0 + (
            sigma_sq / (sigma_bar_sq + sigma_sq)
        ) * x1

        variance_t = (sigma_sq * sigma_bar_sq) / (sigma_bar_sq + sigma_sq)
        std_dev_t = torch.sqrt(variance_t)

        return torch.normal(mean=mu_t, std=std_dev_t)

    def calculate_loss(self, model_output, x0, x1, t):
        if model_output.shape[1] == 6:
            model_output = model_output[:, :3]

        sigma_sq = self._extract(self.sigma_squared, t, x0)
        target = (x1 - x0) / torch.sqrt(sigma_sq)
        return (model_output - target).pow(2).mean()

    def sample_ddpm(self, model, x1, n_steps):
        x_t = x1
        for i in range(n_steps - 1, -1, -100):
            t = torch.tensor([i] * x1.shape[0], device=x1.device)

            sigma_sq = self._extract(self.sigma_squared, t, x1)
            sigma_t = torch.sqrt(sigma_sq)

            model_output = model(x_t, t)
            if model_output.shape[1] == 6:
                model_output = model_output[:, :3]

            x0_pred = x_t - sigma_t * model_output

            if i > 0:
                x_t = self.sample_xt(x0_pred, x_t, t - 1)
            else:
                x_t = x0_pred

        return x_t
