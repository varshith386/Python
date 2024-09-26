from qiskit import QuantumCircuit, transpile, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def custom_execute(qc, shots=1024, backend='qasm_simulator'):
    job = execute(qc, backend, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    return counts

class QuantumSimulator:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.qc = QuantumCircuit(num_qubits)
    
    def add_gate(self, gate, target):
        getattr(self.qc, gate)(target)
    
    def measure_qubits(self):
        self.qc.measure_all()
    
    def simulate(self, shots=1024, backend='qasm_simulator'):
        return custom_execute(self.qc, shots, backend)

class QuantumCircuitEditor:
    def __init__(self, num_qubits):
        self.simulator = QuantumSimulator(num_qubits)
    
    def add_gate(self, gate, target):
        self.simulator.add_gate(gate, target)
    
    def measure_qubits(self):
        self.simulator.measure_qubits()
    
    def simulate_and_visualize(self, shots=1024, backend='qasm_simulator'):
        counts = self.simulator.simulate(shots, backend)
        plot_histogram(counts)
        plt.show()

# Example usage:
editor = QuantumCircuitEditor(num_qubits=2)
editor.add_gate('h', 0)  # Add Hadamard gate to qubit 0
editor.add_gate('cx', 0, 1)  # Add CNOT gate with control qubit 0 and target qubit 1
editor.measure_qubits()  # Measure qubits
editor.simulate_and_visualize()  # Simulate and visualize the quantum circuit
