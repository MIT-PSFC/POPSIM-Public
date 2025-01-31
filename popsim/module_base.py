import inspect
from abc import ABC, abstractmethod
from typing import ClassVar

from jaxtyping import PyTree


class ModuleBase(ABC):
    state_dims: ClassVar[PyTree] = {}
    output_dims: ClassVar[PyTree] = {}

    def __post_init__(self):
        # Derived classes must be registered as PyTrees.
        if not isinstance(self, PyTree):
            raise TypeError(f"{type(self).__name__} must be a PyTree")

        # Derived classes must define the following inner classes.
        required_classes = ["State", "Inputs", "Output"]
        for class_name in required_classes:
            if not hasattr(self, class_name):
                raise TypeError(f"{class_name} must be defined in {type(self).__name__}")

        self._check_call_signature()

    def _check_call_signature(self):
        # Get the __call__ method of the current (sub)class
        call_method = self.__class__.__call__

        # Get the signature of the __call__ method
        signature = inspect.signature(call_method)

        # Define required parameters
        required_inputs = ("state", "inputs")

        # Check if all required parameters are in the signature
        for param_name in required_inputs:
            if param_name not in signature.parameters:
                raise TypeError(f"__call__ method in {self.__class__.__name__} must have a '{param_name}' parameter")

    @abstractmethod
    def __call__(self, state: "State", inputs: "Inputs") -> tuple["State", "Output"]:  # type: ignore # noqa: F821, PGH003
        raise NotImplementedError("This method must be overridden in a subclass.")
