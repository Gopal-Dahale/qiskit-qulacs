"""QulacsBackend class."""
from typing import Iterable, Union, List

from qiskit.providers import BackendV2 as Backend, Options, QubitProperties
from qiskit import QuantumCircuit
from qulacs import QuantumCircuit as Circuit, QuantumState

from .qulacs_job import QulacsJob
from .adapter import qiskit_to_qulacs, local_simulator_to_target


class QulacsBackend(Backend):
    """QulacsBackend class."""

    def __init__(self):
        super().__init__()
        self.name = "statevector_simulator"
        self._target = local_simulator_to_target()

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        return Options()

    @property
    def dtm(self) -> float:
        raise NotImplementedError(
            "System time resolution of output signals",
            f"is not supported by {self.name}.",
        )

    @property
    def meas_map(self) -> List[List[int]]:
        raise NotImplementedError(f"Measurement map is not supported by {self.name}.")

    def qubit_properties(
        self, qubit: Union[int, List[int]]
    ) -> Union[QubitProperties, List[QubitProperties]]:
        raise NotImplementedError

    def drive_channel(self, qubit: int):
        raise NotImplementedError(f"Drive channel is not supported by {self.name}.")

    def measure_channel(self, qubit: int):
        raise NotImplementedError(f"Measure channel is not supported by {self.name}.")

    def acquire_channel(self, qubit: int):
        raise NotImplementedError(f"Acquire channel is not supported by {self.name}.")

    def control_channel(self, qubits: Iterable[int]):
        raise NotImplementedError(f"Control channel is not supported by {self.name}.")

    def run(
        self,
        run_input: Union[QuantumCircuit, List[QuantumCircuit]],
        **options,
    ) -> QulacsJob:
        convert_input = (
            [run_input] if isinstance(run_input, QuantumCircuit) else list(run_input)
        )
        circuits: List[Circuit] = list(qiskit_to_qulacs(convert_input))
        try:
            tasks = zip(
                [QuantumState(circuit.get_qubit_count()) for circuit in circuits],
                circuits,
            )
        except Exception as ex:
            raise ex

        task_id = "id"

        return QulacsJob(
            task_id=task_id,
            tasks=tasks,
            backend=self,
        )

    def __repr__(self):
        return f"QulacsBackend[{self.name}]"
