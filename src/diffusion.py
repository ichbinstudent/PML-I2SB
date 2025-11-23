class DiffusionProcess():
    
    def __init__(self, schedule_name, timesteps):
        self.timesteps = timesteps
        self.schedule_name = schedule_name


    def sample_timesteps(self, batch_size):
        raise NotImplementedError("sample_timesteps method not implemented yet.")
    
    def sample_xt(self, x0, x1, t):
        raise NotImplementedError("sample_xt method not implemented yet.")


    def calculate_loss(self, model_output, target):
        raise NotImplementedError("calculate_loss method not implemented yet.")
    
    def sample_ddpm(self, model, x1):
        raise NotImplementedError("sample_ddpm method not implemented yet.")
    
