from qulacs import QuantumCircuit as Circuit, Observable, PauliOperator
from qiskit import QuantumCircuit
from typing import Iterable, List, Optional, Dict, Union, Tuple
import qulacs.gate as qulacs_gate
from qiskit import extensions as ex
from qiskit.circuit import ParameterExpression, Parameter
from qiskit.circuit.library import Measure
from qulacs import ParametricQuantumCircuit
from qiskit.transpiler import InstructionProperties, Target
import numpy as np
import itertools

qulacs_ops = set(dir(qulacs_gate))

QISKIT_OPERATION_MAP_SELF_ADJOINT = {
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
	"U1": ex.U1Gate,
	"U2": ex.U2Gate,
	"U3": ex.U3Gate,
	"U3": ex.UGate,
	"S": ex.SGate,
	"Sdag": ex.SdgGate,
	"T": ex.TGate,
	"Tdag": ex.TdgGate,
	"sqrtX": ex.SXGate,
	"sqrtXdag": ex.SXdgGate,
	"DenseMatrix": ex.UnitaryGate,
}

QISKIT_OPERATION_MAP_PARAMETRIC = {
	"ParametricRX": ex.RXGate,
	"ParametricRY": ex.RYGate,
	"ParametricRZ": ex.RZGate,
}

QISKIT_GATE_TO_QULACS_GATE_MAPPING = {
	# native PennyLane operations also native to qiskit
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
	"U3": ex.UGate(Parameter("theta"), Parameter("phi"), Parameter("lam")),
	"S": ex.SGate(),
	"Sdag": ex.SdgGate(),
	"T": ex.TGate(),
	"Tdag": ex.TdgGate(),
	"sqrtX": ex.SXGate(),
	"sqrtXdag": ex.SXdgGate(),
}

QISKIT_OPERATION_MAP = {
	**QISKIT_OPERATION_MAP_SELF_ADJOINT,
}

inv_map = {v.__name__: k for k, v in QISKIT_OPERATION_MAP.items()}
inv_parametric_map = {v.__name__: k for k, v in QISKIT_OPERATION_MAP_PARAMETRIC.items()}

def local_simulator_to_target() -> Target:

	target = Target()

	instructions = [
		inst
		for inst in QISKIT_GATE_TO_QULACS_GATE_MAPPING.values()
		if inst is not None
	]

	num_qubits = 26

	target.add_instruction(Measure(), {(i,): None for i in range(num_qubits)})

	for instruction in instructions:
		instruction_props: Optional[
			Dict[Union[Tuple[int], Tuple[int, int]], Optional[InstructionProperties]]
		] = {}

		if instruction.num_qubits == 1:
			for i in range(num_qubits):
				instruction_props[(i,)] = None
			target.add_instruction(instruction, instruction_props)
		elif instruction.num_qubits == 2:
			for src in range(num_qubits):
				for dst in range(num_qubits):
					if src != dst:
						instruction_props[(src, dst)] = None
						instruction_props[(dst, src)] = None
			target.add_instruction(instruction, instruction_props)
	return target

def _extract_variable_refs(parameters, values):
	return dict(zip(parameters, values))

def map_wires(qc_wires):
	return dict(zip(qc_wires, range(len(qc_wires))))


def circuit_mapper(qcirc: QuantumCircuit):

	var_ref_map = _extract_variable_refs(qcirc.parameters, range(qcirc.num_parameters))

	gate_list = []
	wire_list = []
	param_idx_list = []
	param_value_list = []

	qc_wires = [hash(q) for q in qcirc.qubits]
	wire_map = map_wires(qc_wires)

	for op, qargs, cargs in qcirc.data:
		instruction_name = op.__class__.__name__
		operation_wires = [wire_map[hash(qubit)] for qubit in qargs]
		operation_name = None

		if (
			instruction_name in inv_map
			and inv_map[instruction_name] in qulacs_ops
		):
			pl_parameters_idx = []
			pl_parameters_value = []

			for p in op.params:
				if isinstance(p, ParameterExpression):
					if p.parameters:
						ordered_params = tuple(p.parameters)
						f_args = []
						for i_ordered_params in ordered_params:
							f_args.append(var_ref_map.get(i_ordered_params))
						pl_parameters_idx.append(*f_args)
						if instruction_name not in ['RXGate', 'RYGate', 'RZGate']:
							raise ValueError("Only RX, RY, RZ Parametric gates are supported")
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

		else:
			try:
				operation_matrix = op.to_matrix()
				operation_name = inv_map[ex.UnitaryGate]
				parameters_idx = []
				parameters_value = operation_matrix
				wires = operation_wires

				operation = getattr(qulacs_gate, operation_name)

				gate_list.append(operation)
				wire_list.append(wires)
				param_idx_list.append(parameters_idx)
				param_value_list.append(parameters_value)

			except BaseException as e:
				print(e)

	return gate_list, wire_list, param_idx_list, param_value_list


def convert_qiskit_to_qulacs_circuit(circuit: QuantumCircuit) -> Union[Circuit, Dict]:

	gate_list, wire_list, param_idx_list, param_value_list = circuit_mapper(circuit)
	n_qubits = circuit.num_qubits

	def circuit_builder(params: np.ndarray = None):
		circuit = ParametricQuantumCircuit(n_qubits)

		for gate, wires, param_idx, param_value in zip(gate_list, wire_list, param_idx_list, param_value_list):

			if param_idx:
				if gate in [qulacs_gate.U1, qulacs_gate.U2, qulacs_gate.U3]:
					parameters = params[param_idx]
				else:
					parameters = np.negative(params[param_idx])
					circuit.add_parametric_gate(gate(*wires, *parameters))
			else:
				parameters = np.negative(param_value)
				circuit.add_gate(gate(*wires, *parameters))

		return circuit

	# print(param_idx_list)

	metadata = {
		'paramater_mapping': list(itertools.chain.from_iterable(param_idx_list))
	}

	return circuit_builder, metadata


def qiskit_to_qulacs(circuits: List[QuantumCircuit])-> Iterable[Circuit]:
	for circuit in circuits:
		yield convert_qiskit_to_qulacs_circuit(circuit)[0]()

def convert_sparse_pauliop_to_qulacs_observable(sparse_pauliop):
    qulacs_observable = Observable(sparse_pauliop.num_qubits)

    for op in sparse_pauliop:
        term, coefficient = op.paulis[0].__str__()[::-1], op.coeffs[0]

        pauli_string = ""
        for qubit, pauli in enumerate(term):
            pauli_string += f"{pauli} {qubit} "

        qulacs_observable.add_operator(PauliOperator(pauli_string, coefficient))

    return qulacs_observable
