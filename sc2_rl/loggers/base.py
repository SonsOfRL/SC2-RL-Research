from stable_baselines3.common.callbacks import BaseCallback


class LogParametersCallback(BaseCallback):

    def _on_training_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        """
        :return: (bool) If the callback returns False, training is aborted early.
        """
        _self = self.locals["self"]
        _logger = self.globals["logger"].Logger.CURRENT
        for param_name, param in _self.policy.state_dict().items():
            key = "weight/{}, std".format(param_name)
            _logger.record_mean(key, param.std().item())
            key = "weight/{}, mean".format(param_name)
            _logger.record_mean(key, param.mean().item())
        return True