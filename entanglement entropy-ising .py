import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Pauli matrices
sx = np.array([[0,1],[1,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)
id2 = np.eye(2, dtype=complex)

def kron_list(mats):
    """Kronecker product of list of matrices (left to right)."""
    res = mats[0]
    for M in mats[1:]:
        res = np.kron(res, M)
    return res

def build_hamiltonian(N, J, h):
    H = np.zeros((2**N, 2**N), dtype=complex)
    # interaction J sum sz_i sz_{i+1}
    for i in range(N-1):
        ops = [id2]*N
        ops[i]   = sz
        ops[i+1] = sz
        H += J * kron_list(ops)
    # field terms -h sum sz_i and -h sum sx_i (as in your formula)
    for i in range(N):
        ops = [id2]*N
        ops[i] = sz
        H += -h * kron_list(ops)
        ops = [id2]*N
        ops[i] = sx
        H += -h * kron_list(ops)
    return H

def partial_trace(rho, sysA, dims):
    """Partial trace over subsystems not in sysA.
       dims: list of subsystem dimensions (here all 2).
       sysA: list of indices to keep (e.g., [0] for first spin).
    """
    # reshape to tensor with indices (A,B,A',B')
    dimA = np.prod([dims[i] for i in sysA])
    sysB = [i for i in range(len(dims)) if i not in sysA]
    dimB = np.prod([dims[i] for i in sysB])
    rho_reshaped = rho.reshape([dimA, dimB, dimA, dimB])
    # trace over B
    rhoA = np.zeros((dimA, dimA), dtype=complex)
    for i in range(dimB):
        rhoA += rho_reshaped[:, i, :, i]
    return rhoA

def von_neumann_entropy(rho, base=2):
    vals = np.real(la.eigvals(rho))
    vals = np.clip(vals, 0, 1)
    vals = vals[vals>1e-12]
    return -np.sum(vals * np.log(vals) / np.log(base))

# Parameters
N = 5
J = 1.0
hs = np.linspace(0.0, 2.0, 41)  # sweep h
entropies = []

for h in hs:
    H = build_hamiltonian(N, J, h)
    # diagonalize (ground state)
    E, V = la.eigh(H)
    psi0 = V[:, 0]  # ground state vector
    rho = np.outer(psi0, np.conj(psi0))  # pure state density matrix
    rhoA = partial_trace(rho, sysA=[0], dims=[2]*N)  # keep spin 0
    S = von_neumann_entropy(rhoA, base=2)
    entropies.append(S)

# Plot
plt.figure(figsize=(6,4))
plt.plot(hs, entropies, '-o')
plt.xlabel('h')
plt.ylabel('Von Neumann entropy S(spin 0)')
plt.title('Entanglement entropy of spin 0 with spins 1-4 (N=5)')
plt.grid(True)
plt.show()
