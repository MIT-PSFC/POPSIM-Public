import inspect
from abc import ABC, abstractmethod
from typing import ClassVar

import equinox as eqx
from jaxtyping import PyTree


class TimeIndepModule(eqx.Module):
    """For now, no special requirements for time-independent modules."""


class TimeDepModule(eqx.Module, ABC):
    state_dims: ClassVar[PyTree] = {}
    output_dims: ClassVar[PyTree] = {}

    def __check__init__(self):
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
        if self.__class__.__call__ is TimeDepModule.__call__:
            # This check is for concrete implementations.
            # If a subclass doesn't override __call__, the abstractmethod error will be raised upon instantiation.
            # Or, if an instance of TimeDepModule itself were somehow created (which ABC prevents).
            return

        call_method = self.__class__.__call__
        signature = inspect.signature(call_method)
        required_params = ("state", "inputs")

        for param_name in required_params:
            if param_name not in signature.parameters:
                raise TypeError(f"__call__ method in {self.__class__.__name__} must have a '{param_name}' parameter")

    @abstractmethod
    def __call__(self, state: "State", inputs: "Inputs") -> tuple["State", "Output"]:  # type: ignore # noqa: F821, PGH003
        raise NotImplementedError("This method must be overridden in a subclass.")
