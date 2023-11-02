"""Util functions for provider"""
import itertools
from math import log2
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit import extensions as ex
from qiskit import transpile
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.utils import local_hardware_info
from scipy.sparse import csc_matrix

import qulacs.gate as qulacs_gate
from qulacs import Observable, ParametricQuantumCircuit, PauliOperator
from qulacs.gate import U1, U2, U3

_EPS = 1e-10  # global variable used to chop very small numbers to zero

# Available system memory
SYSTEM_MEMORY_GB = local_hardware_info()["memory"]

# Max number of qubits
MAX_QUBITS = int(log2(SYSTEM_MEMORY_GB * (1024**3) / 16))

qulacs_ops = set(dir(qulacs_gate))

# TODO: Add more gates maybe
QISKIT_OPERATION_MAP = {
    # native PennyLane operations also native to qiskit
    "X": ex.XGate,
    "Y": ex.YGate,
    "Z": ex.ZGate,
    "H": ex.HGate,
    "CNOT": ex.CXGate,
    "CZ": ex.CZGate,
    "SWAP": ex.SwapGate,
    "FREDKIN": ex.CSwapGate,
    "RX": ex.RXGate,
    "RY": ex.RYGate,
    "RZ": ex.RZGate,
    "Identity": ex.IGate,
    "TOFFOLI": ex.CCXGate,
    "U1": ex.U1Gate,  # deprecated in qiskit use p gate
    "U2": ex.U2Gate,  # deprecated in qiskit use u gate
    "U3": ex.U3Gate,
    "S": ex.SGate,
    "Sdag": ex.SdgGate,
    "T": ex.TGate,
    "Tdag": ex.TdgGate,
    "sqrtX": ex.SXGate,
    "sqrtXdag": ex.SXdgGate,
    "DenseMatrix": ex.UnitaryGate,
}

QISKIT_GATE_TO_QULACS_GATE_MAPPING = {
    # native qulacs operations also native to qiskit
    "X": ex.XGate(),
    "Y": ex.YGate(),
    "Z": ex.ZGate(),
    "H": ex.HGate(),
    "CNOT": ex.CXGate(),
    "CZ": ex.CZGate(),
    "SWAP": ex.SwapGate(),
    "FREDKIN": ex.CSwapGate(),
    "RX": ex.RXGate(Parameter("theta")),
    "RY": ex.RYGate(Parameter("theta")),
    "RZ": ex.RZGate(Parameter("theta")),
    "Identity": ex.IGate(),
    "TOFFOLI": ex.CCXGate(),
    "U1": ex.U1Gate(Parameter("theta")),
    "U2": ex.U2Gate(Parameter("theta"), Parameter("lam")),
    "U3": ex.U3Gate(Parameter("theta"), Parameter("phi"), Parameter("lam")),
    "S": ex.SGate(),
    "Sdag": ex.SdgGate(),
    "T": ex.TGate(),
    "Tdag": ex.TdgGate(),
    "sqrtX": ex.SXGate(),
    "sqrtXdag": ex.SXdgGate(),
}

QISKIT_OPERATION_MAP_PARAMETRIC = {
    "ParametricRX": ex.RXGate,
    "ParametricRY": ex.RYGate,
    "ParametricRZ": ex.RZGate,
}

inv_map = {v.__name__: k for k, v in QISKIT_OPERATION_MAP.items()}
inv_parametric_map = {v.__name__: k for k, v in QISKIT_OPERATION_MAP_PARAMETRIC.items()}

translatable_qiskit_gates = set(
    map(lambda gate: gate.name, list(QISKIT_GATE_TO_QULACS_GATE_MAPPING.values()))
)


def _extract_variable_refs(parameters, values):
    """
    The function `_extract_variable_refs` takes two lists, `parameters` and
    `values`, and returns a dictionary where the elements of `parameters`
    are the keys and the elements of `values` are the values.

    Args:
      parameters: A list of variable names or parameter names.
      values: A list of values that correspond to the parameters.

    Returns:
      a dictionary that maps each parameter to its corresponding value.
    """
    return dict(zip(parameters, values))


def map_wires(qc_wires: List[int]):
    """
    The function `map_wires` takes a list of quantum circuit wires and returns
    a dictionary mapping each wire to its corresponding index.

    Args:
      qc_wires: A list of quantum circuit wires.

    Returns:
      The function `map_wires` returns a dictionary where the keys are the
      elements of `qc_wires` and the values are the corresponding indices
      of the elements in `qc_wires`.
    """
    return dict(zip(qc_wires, range(len(qc_wires))))


def circuit_mapper(qcirc: QuantumCircuit):
    """
    The function `circuit_mapper` takes a QuantumCircuit object as input and
    extracts information about the gates, wires, and parameters in the circuit.

    Args:
      qcirc (QuantumCircuit): The `qcirc` parameter is of type `QuantumCircuit`
      . It represents a quantum circuit, which is a collection of quantum
      gates and measurements that can be applied to a set of qubits.

    Returns:
      The function `circuit_mapper` returns four lists: `gate_list`,
      `wire_list`, `param_idx_list`, and `param_value_list`. These lists
      contain information about the gates, wires, parameter indices, and
      parameter values extracted from the input `qcirc` QuantumCircuit.
    """
    gate_names = {gate.name for gate, _, _ in qcirc.data}

    if "barrier" in gate_names:
        qcirc = RemoveBarriers()(qcirc)
        gate_names.remove("barrier")

    if not gate_names.issubset(translatable_qiskit_gates):
        qcirc = transpile(qcirc, basis_gates=translatable_qiskit_gates)

    var_ref_map = _extract_variable_refs(qcirc.parameters, range(qcirc.num_parameters))

    gate_list = []
    wire_list = []
    param_idx_list = []
    param_value_list = []

    qc_wires = [hash(q) for q in qcirc.qubits]
    wire_map = map_wires(qc_wires)

    for op, qargs, _ in qcirc.data:
        instruction_name = op.__class__.__name__

        operation_wires = [wire_map[hash(qubit)] for qubit in qargs]
        operation_name = None

        pl_parameters_idx: List[List[int]] = []
        pl_parameters_value = []

        for p in op.params:
            if isinstance(p, ParameterExpression):
                if p.parameters:
                    ordered_params = tuple(p.parameters)
                    f_args = []
                    for i_ordered_params in ordered_params:
                        f_args.append(var_ref_map.get(i_ordered_params))
                    pl_parameters_idx.append(*f_args)
                    if instruction_name not in [
                        "RXGate",
                        "RYGate",
                        "RZGate",
                    ]:
                        raise ValueError(
                            "RX, RY, RZ Parametric gates are supported",
                            "for now.",
                        )
                    operation_name = inv_parametric_map[instruction_name]
                else:
                    pl_parameters_value.append(float(p))
            else:
                pl_parameters_value.append(float(p))

        if operation_name is None:
            operation_name = inv_map[instruction_name]

        parameters_idx = pl_parameters_idx
        parameters_value = pl_parameters_value
        wires = operation_wires
        operation = getattr(qulacs_gate, operation_name)
        gate_list.append(operation)
        wire_list.append(wires)
        param_idx_list.append(parameters_idx)
        param_value_list.append(parameters_value)

    return gate_list, wire_list, param_idx_list, param_value_list, qcirc.global_phase


def convert_qiskit_to_qulacs_circuit(
    circuit: QuantumCircuit,
) -> Tuple[Callable[[np.ndarray], ParametricQuantumCircuit], Dict]:
    """
    The function `convert_qiskit_to_qulacs_circuit` converts a Qiskit circuit
    to a Qulacs circuit, including parameter mapping.

    Args:
      circuit (QuantumCircuit): The `circuit` parameter is a Qiskit
      `QuantumCircuit` object. It represents a quantum circuit that you want
      to convert to a Qulacs circuit.

    Returns:
      The function `convert_qiskit_to_qulacs_circuit` returns a tuple
      containing two elements:
    """

    gate_list, wire_list, param_idx_list, param_value_list, gphase = circuit_mapper(
        circuit
    )
    n_qubits = circuit.num_qubits

    def circuit_builder(params: np.ndarray = None):
        circuit = ParametricQuantumCircuit(n_qubits)

        for gate, wires, param_idx, param_value in zip(
            gate_list, wire_list, param_idx_list, param_value_list
        ):
            # Gates that need negative params
            # Rx, Ry, Rz
            # Change if-else

            if param_idx:
                if gate in [U1, U2, U3]:
                    parameters = params[param_idx]
                else:
                    # print("Negating parameters")
                    parameters = np.negative(params[param_idx])
                    circuit.add_parametric_gate(gate(*wires, *parameters))
            else:
                if gate in [U1, U2, U3]:
                    parameters = param_value
                else:
                    parameters = np.negative(param_value)
                circuit.add_gate(gate(*wires, *parameters))

        if gphase > _EPS:
            gphase_mat = csc_matrix((2**n_qubits, 2**n_qubits), dtype=complex)

            # set the diagonal to be the global phase
            gphase_mat.setdiag(np.exp(1j * gphase))

            # add the gphase_mat to the circuit
            circuit.add_gate(
                qulacs_gate.SparseMatrix(  # pylint: disable=no-member
                    list(np.arange(0, n_qubits, 1)), gphase_mat
                )
            )

        return circuit

    metadata = {
        "paramater_mapping": list(itertools.chain.from_iterable(param_idx_list))
    }

    return circuit_builder, metadata


def qiskit_to_qulacs(
    circuits: List[QuantumCircuit],
) -> Iterable[ParametricQuantumCircuit]:
    """
    The function `qiskit_to_qulacs` converts a list of Qiskit quantum circuits
    into a generator of Qulacs circuits.

    Args:
      circuits (List[QuantumCircuit]): The `circuits` parameter is a list of
      `QuantumCircuit` objects.
    """
    for circuit in circuits:
        yield convert_qiskit_to_qulacs_circuit(circuit)[0]()  # type: ignore


def convert_sparse_pauliop_to_qulacs_obs(sparse_pauliop: SparsePauliOp):
    """
    The function `convert_sparse_pauliop_to_qulacs_obs` converts a
    sparse Pauli operator to a Qulacs observable.

    Args:
      sparse_pauliop: The `sparse_pauliop` parameter is a sparse
      representation of a Pauli operator. It is an object that contains
      information about the Pauli terms and their coefficients. Each term is
      represented by a `PauliTerm` object, which consists of a list of Pauli
      operators and their corresponding

    Returns:
      a Qulacs Observable object.
    """
    qulacs_observable = Observable(sparse_pauliop.num_qubits)

    for op in sparse_pauliop:
        term, coefficient = str(op.paulis[0])[::-1], op.coeffs[0]

        pauli_string = ""
        for qubit, pauli in enumerate(term):
            pauli_string += f"{pauli} {qubit} "

        qulacs_observable.add_operator(PauliOperator(pauli_string, coefficient.real))

    return qulacs_observable
