# Qiskit-Qulacs Installation Guide

This document provides step-by-step instructions to set up the Python environment, install dependencies, and install the Qiskit-Qulacs plugin.

Setting up Python Environment
## Setting up Python Environment

Create a fresh Python environment using your preferred method (e.g., virtualenv, conda).

```
virtualenv qiskit-qulacs-env
source qiskit-qulacs-env/bin/activate
```

Ensure that you have a compatible version of Python (e.g., Python 3.7 or higher) installed in the environment.

## Installing Qiskit-Qulacs Software

```
git clone https://github.com/Gopal-Dahale/qiskit-qulacs.git
cd qiskit-qulacs
pip install -r requirements.txt
pip install .
```

## Testing the Installation

Since Qiskit-Qulacs is currently under heavy development, there are no tests yet. However, you can test the installation using a simple program:

```python
import qiskit
from qiskit_qulacs import QulacsProvider

# Create a quantum circuit
qc = qiskit.QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Use Qiskit-Qulacs to run the circuit
backend = QulacsProvider().get_backend('statevector_simulator')
result = backend.run(qc).result()

# Get the statevector
statevector = result.get_statevector()

# Print the statevector
print(statevector)
```
