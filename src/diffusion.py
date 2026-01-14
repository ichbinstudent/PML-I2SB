import torch


class DiffusionProcess:

    def __init__(self, beta_schedule):
        self.beta = beta_schedule
        self.n_steps = len(beta_schedule)
        self.t = torch.linspace(0, 1, self.n_steps)

        dt = self.t[1] - self.t[0]

        self.std_fwd_sq = torch.cumsum(self.beta, dim=0) * dt
        self.std_fwd = torch.sqrt(self.std_fwd_sq)
        
        self.std_bwd_sq = torch.flip(torch.cumsum(torch.flip(self.beta, [0]), dim=0), [0]) * dt
        self.std_bwd = torch.sqrt(self.std_bwd_sq)

        denom = self.std_fwd**2 + self.std_bwd**2
        self.mu_x0 = self.std_bwd**2 / denom
        self.mu_x1 = self.std_fwd**2 / denom

        self.std_sb = torch.sqrt((self.std_fwd**2 * self.std_bwd**2) / denom)

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.n_steps, (batch_size,))

    @staticmethod
    def _extract(arr, t, x, dtype=torch.float, device=torch.device("cpu"), ndim=4):
        if x is not None:
            dtype = x.dtype
            device = x.device
            ndim = x.ndim
        out = torch.as_tensor(arr, dtype=dtype, device=device).gather(0, t)
        return out.reshape((-1,) + (1,) * (ndim - 1))

    def sample_xt(self, x0, x1, t):
        """
        From equation (11) in the paper

        X_t = N(mu_t, sigma_t^2 I)
        mu_t = (sigma_bar_t^2 / (sigma_bar_t^2 + sigma_t^2)) * X_0 + (sigma_t^2 / (sigma_bar_t^2 + sigma_t^2)) * X_1
        Sigma_t = sigma_t^2 * sigma_bar_t^2 / (sigma_bar_t^2 + sigma_t^2)
        """
        std_sb = self._extract(self.std_sb, t, x0)
        mu_x0 = self._extract(self.mu_x0, t, x0)
        mu_x1 = self._extract(self.mu_x1, t, x0)
        
        xt = mu_x0 * x0 + mu_x1 * x1
        xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def posterior(self, x0_pred, x_n, n, nprev):
        """
        From equation (4) in the paper

        p(x_n | x_0, x_{n + 1}) = N(mu_post, var_post I)
        """
        std_n = self._extract(self.std_fwd, n, x_n)
        std_nprev = self._extract(self.std_fwd, nprev, x_n)
        std_delta = torch.sqrt(torch.clamp(std_n**2 - std_nprev**2, min=1e-10))
        
        denom = std_nprev**2 + std_delta**2
        mu_x0 = std_delta**2 / denom
        mu_xn = std_nprev**2 / denom
        var = (std_nprev**2 * std_delta**2) / denom
        
        xt_prev = mu_x0 * x0_pred + mu_xn * x_n
        
        if nprev[0].item() > 0:
            xt_prev = xt_prev + torch.sqrt(var) * torch.randn_like(xt_prev)
        
        return xt_prev

    def calculate_loss(self, model_output, x0, xt, t):
        """
        From equation (12) in the paper

        Model predicts the label: (x_t - x_0) / std_fwd
        """
        if model_output.shape[1] == 6:
            model_output = model_output[:, :3]

        std_fwd = self._extract(self.std_fwd, t, x0)
        target = ((xt - x0) / std_fwd).detach()
        return (model_output - target).pow(2).mean()

    @torch.no_grad()
    def sample_ddpm(self, model, x1, n_steps, precision: float = 1) -> torch.Tensor:
        """
        Sample using I2SB reverse process (Algorithm 2 from the paper).
        """
        assert 0 < precision <= 1.0, "precision must be in (0, 1]"

        step_size = max(1, int(1.0 / precision))
        steps = list(range(0, n_steps, step_size))

        if steps[-1] != n_steps - 1:
            steps.append(n_steps - 1)
        
        steps = steps[::-1]
        
        xt = x1.detach()
        
        for prev_step, step in zip(steps[1:], steps[:-1]):
            t = torch.full((x1.shape[0],), step, device=x1.device, dtype=torch.long)
            t_prev = torch.full((x1.shape[0],), prev_step, device=x1.device, dtype=torch.long)
            
            std_fwd = self._extract(self.std_fwd, t, xt)

            model_pred = model(xt, t)
            if model_pred.shape[1] == 6:
                model_pred = model_pred[:, :3]
            
            x0_pred = xt - std_fwd * model_pred
            xt = self.posterior(x0_pred, xt, t, t_prev)
        
        return xt
