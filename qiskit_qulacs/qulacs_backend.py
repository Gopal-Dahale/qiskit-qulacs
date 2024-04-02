"""QulacsBackend class."""

import copy
import time
import uuid
from collections import Counter
from typing import List, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import BackendV1 as Backend
from qiskit.providers import JobStatus, Options
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData

import qulacs
from qulacs.circuit import QuantumCircuitOptimizer

from .adapter import MAX_QUBITS, qiskit_to_qulacs
from .backend_utils import BASIS_GATES, available_devices, generate_config
from .qulacs_job import QulacsJob
from .version import __version__


class QulacsBackend(Backend):
    """QulacsBackend class."""

    _BASIS_GATES = BASIS_GATES

    _DEFAULT_CONFIGURATION = {
        "backend_name": "qulacs_simulator",
        "backend_version": __version__,
        "n_qubits": MAX_QUBITS,
        "url": "https://github.com/Gopal-Dahale/qiskit-qulacs",
        "simulator": True,
        "local": True,
        "conditional": True,
        "open_pulse": False,
        "memory": True,
        "max_shots": int(1e6),
        "description": "A Qulacs fast quantum circuit simulator",
        "coupling_map": None,
        "basis_gates": _BASIS_GATES,
        "gates": [],
    }

    _SIMULATION_DEVICES = ("CPU", "GPU")

    _AVAILABLE_DEVICES = None

    def __init__(
        self, configuration=None, properties=None, provider=None, **backend_options
    ):
        # Update available devices for class
        if QulacsBackend._AVAILABLE_DEVICES is None:
            QulacsBackend._AVAILABLE_DEVICES = available_devices(
                QulacsBackend._SIMULATION_DEVICES
            )

        # Default configuration
        if configuration is None:
            configuration = QasmBackendConfiguration.from_dict(
                QulacsBackend._DEFAULT_CONFIGURATION
            )

        super().__init__(
            configuration,
            provider=provider,
        )

        # Initialize backend properties
        self._properties = properties

        # Set options from backend_options dictionary
        if backend_options is not None:
            self.set_options(**backend_options)

        # Quantum circuit optimizer (if needed)
        self.qc_opt = QuantumCircuitOptimizer()

        self.class_suffix = {
            "GPU": "Gpu",
            "CPU": "",
        }

    @classmethod
    def _default_options(cls):
        return Options(
            # Global options
            shots=0,
            device="CPU",
            seed_simulator=None,
            # Quantum Circuit Optimizer options
            qco_enable=False,
            qco_method="light",
            qco_max_block_size=2,
        )

    def __repr__(self):
        """String representation of an QulacsBackend."""
        name = self.__class__.__name__
        display = f"'{self.name()}'"
        return f"{name}({display})"

    def available_devices(self):
        """Return the available simulation methods."""
        return copy.copy(self._AVAILABLE_DEVICES)

    def _execute_circuits_job(self, circuits, states, run_options, job_id=""):
        """Run a job"""

        shots = run_options.shots
        seed = (
            run_options.seed_simulator
            if run_options.seed_simulator
            else np.random.randint(1000)
        )

        # Start timer
        start = time.time()

        expt_results = []
        if shots:
            for state, circuit in zip(states, circuits):
                circuit.update_quantum_state(state)
                n = circuit.get_qubit_count()

                samples = state.sampling(shots, seed)
                bitstrings = [format(x, f"0{n}b") for x in samples]
                counts = dict(Counter(bitstrings))

                expt_results.append(
                    ExperimentResult(
                        shots=shots,
                        success=True,
                        status=JobStatus.DONE,
                        data=ExperimentResultData(counts=counts, memory=bitstrings),
                    )
                )
        else:
            for state, circuit in zip(states, circuits):
                circuit.update_quantum_state(state)
                # Statevector
                expt_results.append(
                    ExperimentResult(
                        shots=shots,
                        success=True,
                        status=JobStatus.DONE,
                        data=ExperimentResultData(
                            statevector=state.get_vector(),
                        ),
                    )
                )

        return Result(
            backend_name=self.name(),
            backend_version=self.configuration().backend_version,
            job_id=job_id,
            qobj_id=0,
            success=True,
            results=expt_results,
            status=JobStatus.DONE,
            time_taken=time.time() - start,
        )

    def run(
        self,
        run_input: Union[QuantumCircuit, List[QuantumCircuit]],
        **run_options,
    ) -> QulacsJob:
        run_input = [run_input] if isinstance(run_input, QuantumCircuit) else run_input

        run_input = list(qiskit_to_qulacs(run_input))
        config = (
            generate_config(self.options, run_options) if run_options else self.options
        )

        # Use GPU if available
        if config.device not in self.available_devices():
            if config.device == "GPU":
                raise ValueError("GPU support not installed. Install qulacs-gpu.")
            raise ValueError(f"Device {config.device} not found.")

        class_name = f'QuantumState{self.class_suffix.get(config.device, "")}'
        state_class = getattr(qulacs, class_name)

        # Use Quantum Circuit Optimizer
        if config.qco_enable:
            if config.qco_method == "light":
                for circuit in run_input:
                    self.qc_opt.optimize_light(circuit)
            elif config.qco_method == "greedy":
                for circuit in run_input:
                    self.qc_opt.optimize(circuit, config.qco_max_block_size)

        # Create quantum states
        states = [state_class(circuit.get_qubit_count()) for circuit in run_input]

        # Submit job
        job_id = str(uuid.uuid4())
        qulacs_job = QulacsJob(
            self,
            job_id,
            self._execute_circuits_job,
            circuits=run_input,
            states=states,
            run_options=config,
        )
        qulacs_job.submit()
        return qulacs_job
