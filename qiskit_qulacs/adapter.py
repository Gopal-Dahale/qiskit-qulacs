"""Util functions for provider"""

import re
import warnings
from math import log2
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
import psutil
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.circuit import library as lib
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.quantum_info import SparsePauliOp
from scipy.sparse import diags
from sympy import lambdify

import qulacs.gate as qg
from qulacs import Observable, ParametricQuantumCircuit, PauliOperator

_EPS = 1e-10  # global variable used to chop very small numbers to zero

# Available system memory
SYSTEM_MEMORY_GB = psutil.virtual_memory().total / (1024**3)

# Max number of qubits
MAX_QUBITS = int(log2(SYSTEM_MEMORY_GB * (1024**3) / 16))


# Defintions of some gates that are not directly defined in qulacs
# The `args` argument is of the form *qubits, *parameters
# The gates defined below currently support only single parameter only


def qgUnitary(*args):
    """
    The function `qgUnitary` takes qubits and parameters as input and returns a dense matrix.

    Returns:
      The function `qgUnitary` is returning a `qg.DenseMatrix` object created with the provided `qubits`
    and `parameters`.
    """
    qubits = args[:-1]
    parameters = args[-1]
    return qg.DenseMatrix(qubits, parameters)  # pylint: disable=no-member


IsingXX = lambda *args: qg.ParametricPauliRotation(args[:-1], [1, 1], args[-1].real)
IsingYY = lambda *args: qg.ParametricPauliRotation(args[:-1], [2, 2], args[-1].real)
IsingZZ = lambda *args: qg.ParametricPauliRotation(args[:-1], [3, 3], args[-1].real)

ecr_mat = np.array(
    [[0, 1, 0, 1j], [1, 0, -1j, 0], [0, 1j, 0, 1], [-1j, 0, 1, 0]]
) / np.sqrt(2)

qgECR = lambda *args: qg.DenseMatrix(args, matrix=ecr_mat)

# These gates in qulacs have positive rotation directions.
# Angles of these gates need to be multiplied by -1 during conversion.
# https://docs.qulacs.org/en/latest/guide/2.0_python_advanced.html#1-qubit-rotating-gate

neg_gates = {"RXGate", "RYGate", "RZGate", "RXXGate", "RYYGate", "RZZGate"}

# Only these gates support trainable parameters
parametric_gates = neg_gates

# Gate addition type
# based on the type of the, one of these two will be used in the qulacs circuit
gate_addition = ["add_gate", "add_parametric_gate"]


QISKIT_OPERATION_MAP = {
    qg.X: lib.XGate,
    qg.Y: lib.YGate,
    qg.Z: lib.ZGate,
    qg.H: lib.HGate,
    qg.CNOT: lib.CXGate,
    qg.CZ: lib.CZGate,
    qg.SWAP: lib.SwapGate,
    qg.FREDKIN: lib.CSwapGate,
    qg.ParametricRX: lib.RXGate,  # -theta
    qg.ParametricRY: lib.RYGate,  # -theta
    qg.ParametricRZ: lib.RZGate,  # -theta
    qg.Identity: lib.IGate,
    qg.TOFFOLI: lib.CCXGate,
    qg.U1: lib.U1Gate,  # deprecated in qiskit, use p gate
    qg.U2: lib.U2Gate,  # deprecated in qiskit, use u gate
    qg.U3: lib.U3Gate,  # deprecated in qiskit, use u gate
    IsingXX: lib.RXXGate,  # -theta
    IsingYY: lib.RYYGate,  # -theta
    IsingZZ: lib.RZZGate,  # -theta
    qg.S: lib.SGate,
    qg.Sdag: lib.SdgGate,
    qg.T: lib.TGate,
    qg.Tdag: lib.TdgGate,
    qg.sqrtX: lib.SXGate,
    qg.sqrtXdag: lib.SXdgGate,
    qgUnitary: lib.UnitaryGate,
    qgECR: lib.ECRGate,
}
inv_map = {v.__name__: k for k, v in QISKIT_OPERATION_MAP.items()}

# Gates with different names but same operation
duplicate_gates = {"UGate": "U3Gate"}


def convert_qiskit_to_qulacs_circuit(qc: QuantumCircuit):
    """
    The function `convert_qiskit_to_qulacs_circuit` converts a Qiskit QuantumCircuit to a Qulacs
    ParametricQuantumCircuit while handling parameter mapping and gate operations.

    Args:
      qc (QuantumCircuit): The `qc` is expected to be a QuantumCircuit object from Qiskit.

    Returns:
      The `convert_qiskit_to_qulacs_circuit` function returns a nested function `circuit_builder` that
    takes an optional `params_values` argument. Inside `circuit_builder`, it constructs a
    `ParametricQuantumCircuit` based on the input `QuantumCircuit` `qc` provided to the outer function.

    """

    def circuit_builder(params_values=[]) -> Tuple[ParametricQuantumCircuit, Dict]:
        """
        The `circuit_builder` function converts a Qiskit quantum circuit into a ParametricQuantumCircuit,
        handling parameter mapping and supporting trainable parameters.

        Args:
          params_values: The `params_values` parameter in the `circuit_builder` function is a list that
        contains the values of the parameters that will be used to build a quantum circuit. These values
        will be used to replace the symbolic parameters in the quantum circuit with concrete numerical
        values during the circuit construction process.

        Returns:
          The `circuit_builder` function returns a ParametricQuantumCircuit and a dictionary containing
        information about parameter mapping and parameter expressions.
        """
        circuit = ParametricQuantumCircuit(qc.num_qubits)

        # parameter mapping
        # dictionary from qiskit's quantum circuit parameters to a two element tuple.
        # the tuple has an element params_values and its index
        # Currently not supporting qiskit's parameter expression
        var_ref_map = dict(
            zip(qc.parameters, list(zip(params_values, range(qc.num_parameters)))),
        )

        # Wires from a qiskit circuit have unique IDs, so their hashes are unique too
        qc_wires = [hash(q) for q in qc.qubits]
        wire_map = dict(zip(qc_wires, range(len(qc_wires))))

        # Holds the indices of parameter as they occur during
        # circuit conversion. This is used during circuit gradient computation.
        param_mapping = []
        param_exprs = []

        f_args: List[Any] = []
        f_params: List[Any] = []
        indices: List[int] = []
        f_param_names: Set[Any] = set()
        flag = False  # indicates whether the instruction is parametric

        for instruction, qargs, _ in qc.data:

            # the new Singleton classes have different names than the objects they represent,
            # but base_class.__name__ still matches
            instruction_name = getattr(
                instruction, "base_class", instruction.__class__
            ).__name__

            instruction_name = duplicate_gates.get(instruction_name, instruction_name)

            sign = 1.0 - 2 * (instruction_name in neg_gates)
            operation_wires = [wire_map[hash(qubit)] for qubit in qargs]

            operation_params = []
            flag = False

            for p in instruction.params:
                if isinstance(p, ParameterExpression) and p.parameters:
                    f_args = []
                    f_params = []
                    indices = []

                    # Ensure duplicate subparameters are only appended once.
                    f_param_names = set()

                    for subparam in p.parameters:

                        try:
                            parameter = subparam
                            argument, index = var_ref_map.get(subparam)
                        except:
                            raise ValueError(
                                "The number of circuit parameters does not match",
                                " the number of parameter values passed.",
                            )

                        if isinstance(subparam, ParameterVectorElement):
                            # Unfortunately, the names of parameter vector elements
                            # include square brackets, making them invalid Python
                            # identifiers and causing compatibility problems with SymPy.
                            # To solve this issue, we generate a temporary parameter
                            # that replaces square bracket by underscores.
                            subparam_name = re.sub(r"\[|\]", "_", str(subparam))
                            parameter = Parameter(subparam_name)
                            argument, index = var_ref_map.get(subparam)

                            # Update the subparam in `p`
                            p = p.assign(subparam, parameter)

                        if parameter.name not in f_param_names:
                            f_param_names.add(parameter.name)
                            f_params.append(parameter)
                            f_args.append(argument)
                            indices.append(index)

                    f_expr = getattr(p, "_symbol_expr")

                    if isinstance(p, Parameter):
                        # If `p` is an instance of `Parameter` then we can
                        # we do not need to calculate the expression value
                        operation_params += list(map(lambda x: x * sign, f_args))
                    else:
                        # Calculate the expression value using sympy
                        f = lambdify(f_params, f_expr)
                        operation_params += [f(*f_args) * sign]

                    param_mapping += indices
                    param_exprs += [(f_params, f_expr)]
                    flag = True
                else:
                    operation_params.append(p * sign)

            operation_class = inv_map.get(instruction_name)
            try:
                getattr(circuit, gate_addition[flag])(
                    operation_class(*operation_wires, *operation_params)  # type: ignore
                )
            except:
                if flag:
                    raise ValueError(
                        f"{__name__}: The {instruction_name} does not support trainable parameter.",
                        f" Consider decomposing {instruction_name} into {parametric_gates}.",
                    )

                warnings.warn(
                    f"{__name__}: The {instruction_name} instruction is not supported"
                    " by Qiskit-Qulacs and has not been added to the circuit.",
                    UserWarning,
                )

        if qc.global_phase > _EPS:
            # add the gphase_mat to the circuit
            circuit.add_gate(
                qg.SparseMatrix(  # pylint: disable=no-member
                    list(range(qc.num_qubits)),
                    diags(np.exp(1j * qc.global_phase) * np.ones(2**qc.num_qubits)),
                )
            )

        return circuit, {
            "parameter_mapping": param_mapping,
            "parameter_exprs": param_exprs,
        }

    return circuit_builder


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
        yield convert_qiskit_to_qulacs_circuit(circuit)()[0]


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
