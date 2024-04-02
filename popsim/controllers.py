import chex


@chex.dataclass
class PIController:
    """Propotional-Integral controller in state-space form. The derivative term is missing
    as it is difficult to include in the state-space form."""

    @chex.dataclass
    class Config:
        kp: float
        ki: float

    @chex.dataclass
    class State:
        integral: float = 0.0

    config: Config

    def __call__(self, state: State, error: float) -> tuple[State, float]:
        state_dot = PIController.State(integral=state.integral + error)
        control = self.config.kp * error + self.config.ki * state.integral
        return state_dot, control
