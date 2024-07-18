from abc import ABC, abstractmethod

from jaxtyping import PyTree


class ModuleBase(ABC):
    state_dims = {}
    output_dims = {}

    def __post_init__(self):
        # Derived classes must be registered as PyTrees.
        if not isinstance(self, PyTree):
            raise TypeError(f"{type(self).__name__} must be a PyTree")

        # Derived classes must define the following inner classes.
        required_classes = ["State", "Params", "Output"]
        for class_name in required_classes:
            if not hasattr(self, class_name):
                raise TypeError(f"{class_name} must be defined in {type(self).__name__}")

    @abstractmethod
    def __call__(self, state: "State", params: "Params") -> tuple["State", "Output"]:  # type: ignore # noqa: F821, PGH003
        raise NotImplementedError("This method must be overridden in a subclass.")
