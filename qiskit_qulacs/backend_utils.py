"""
Qulacs simulator backend utils
"""

from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from qiskit.providers import JobError
from qiskit.providers.options import Options

import qulacs

DEFAULT_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def requires_submit(func):
    """
    Decorator to ensure that a submit has been performed before
    calling the method.

    Args:
                    func (callable): test function to be decorated.

    Returns:
                    callable: the decorated function.
    """

    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        if self._future is None:
            raise JobError("Job not submitted yet!. You have to .submit() first!")
        return func(self, *args, **kwargs)

    return _wrapper


# write in the same order as QISKIT_OPERATION_MAP
BASIS_GATES = sorted(
    [
        "x",
        "y",
        "z",
        "h",
        "cx",
        "swap",
        "cswap",
        "rx",
        "ry",
        "rz",
        "id",
        "ccx",
        "u1",
        "u2",
        "u3",
        "rxx",
        "ryy",
        "rzz",
        "s",
        "sdg",
        "t",
        "tdg",
        "sx",
        "sxdg",
        "unitary",
        "ecr",
    ]
)

# Only CPU and GPU states are supported currently
# used in the function `available_devices`
states = {"CPU": "QuantumState", "GPU": "QuantumStateGpu"}


def available_devices(devices):
    """Check available simulation devices by running a dummy circuit."""
    valid_devices = []
    for device in devices:
        state = states.get(device)
        if state:
            try:
                getattr(qulacs, state)(1)
                valid_devices.append(device)
            except AttributeError:
                pass

    return tuple(valid_devices)


def generate_config(backend_options: Options, run_options):
    """generates a configuration to run simulation"""
    config = Options()
    config.update_options(**backend_options)
    config.update_options(**run_options)
    return config
