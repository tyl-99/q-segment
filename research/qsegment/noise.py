from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider as fp

from qsegment import circuits as cx, metrics as mt

BACKENDS = {"sherbrooke": fp.FakeSherbrooke, "fez": fp.FakeFez, "marrakesh": fp.FakeMarrakesh}


def get_backend(name):
    return BACKENDS[name]()


def simulator(backend, method="density_matrix"):
    return AerSimulator.from_backend(backend, method=method)


def measured_qubits(transpiled):
    """Physical qubit that is measured into classical bit k, listed for k = 0, 1, ..."""
    mapping = {}
    for inst in transpiled.data:
        if inst.operation.name == "measure":
            mapping[transpiled.find_bit(inst.clbits[0]).index] = transpiled.find_bit(inst.qubits[0]).index
    return [mapping[k] for k in sorted(mapping)]


def noisy_probs(sim, transpiled, gammas, betas, shots, seed, n_qubits):
    bound = transpiled.assign_parameters(dict(zip(transpiled.parameters, cx.bind_values(transpiled, gammas, betas))))
    counts = sim.run(bound, shots=shots, seed_simulator=seed).result().get_counts()
    return mt.counts_to_probs(counts, n_qubits)
