from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_vocab: int = 50257
    n_positions: int = 1024
    max_steps: int = 2000
    batch_size: int = 32

    def num_parameters(self, include_embedding=False, include_bias_params=True) -> int:
        d_model = self.d_model
        params = 0
        if include_embedding:
            params += self.n_vocab * d_model  # embedding
            params += self.n_positions * d_model  # positional (not trainable)

        layer_params = 0
        d_attn = self.d_model
        layer_params += 3 * (d_model * d_attn + (d_attn if include_bias_params else 0))  # QKV matrices
        layer_params += d_attn * d_model + (d_model if include_bias_params else 0)  # project back to model space
        d_ff = 4 * d_model
        layer_params += 2 * d_model * d_ff + (d_ff + d_model if include_bias_params else 0)  # feed forward
        layer_params += 4 * d_model  # two layer norms
        params += self.n_layers * layer_params

        params += 2 * d_model  # final layer norm
        return params

    def num_activations(self) -> int:
        return 2 * self.batch_size * self.n_positions * self.d_model * (self.n_layers + 1)

    def gpu_memory_gb(self, *, fp16: bool = False) -> float:
        bytes_per_el = 2 if fp16 else 4
        total_bytes = bytes_per_el * (3 * self.num_parameters(include_embedding=True) + self.num_activations())
        return total_bytes / (1024 ** 3)  # GiB

    def flops_per_token(self):
        d_attn = self.d_model
        return 2 * self.num_parameters(include_embedding=False, include_bias_params=True) + 2 * self.n_layers * self.n_positions * d_attn

    def flops_per_step(self):
        return self.batch_size * self.n_positions * self.flops_per_token()


@dataclass
class TrainingResults:
    config: ModelConfig
    train_loss: list[tuple[int, float]]  # list of (step, loss)
    eval_loss: list[tuple[int, float]]  # list of (step, eval_loss)

    @classmethod
    def from_dict(cls, data: dict):
        data['config'] = ModelConfig(**data['config'])
        return cls(**data)
