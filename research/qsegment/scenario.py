SYSTEMS = ["nurse", "mri", "portal", "email", "pacs", "pharm", "ehr", "billing"]
FIXED = 0                                   # nurse, fixed in zone 0 because zone labels are interchangeable
FREE = list(range(1, len(SYSTEMS)))         # indices of the 7 free systems

TRAFFIC = {("nurse", "ehr"): 6, ("nurse", "email"): 3, ("nurse", "pharm"): 2, ("mri", "pacs"): 6,
           ("pacs", "ehr"): 5, ("portal", "ehr"): 4, ("portal", "billing"): 3, ("ehr", "billing"): 4,
           ("pharm", "ehr"): 5, ("billing", "email"): 2, ("portal", "email"): 2}
RISK = {("nurse", "billing"): 3, ("nurse", "pacs"): 2, ("nurse", "ehr"): 7, ("mri", "ehr"): 5,
        ("mri", "billing"): 4, ("mri", "pharm"): 2, ("portal", "ehr"): 6, ("portal", "billing"): 5,
        ("email", "ehr"): 3, ("email", "billing"): 3, ("email", "pharm"): 2, ("ehr", "billing"): 3}

_INDEX = {name: i for i, name in enumerate(SYSTEMS)}


def _lookup(table, i, j):
    a, b = SYSTEMS[i], SYSTEMS[j]
    return table.get((a, b), table.get((b, a), 0))


def traffic(i, j):
    return _lookup(TRAFFIC, i, j)


def risk(i, j):
    return _lookup(RISK, i, j)


PAIRS = sorted({tuple(sorted((_INDEX[a], _INDEX[b]))) for a, b in list(TRAFFIC) + list(RISK)})


def net_weight(i, j):
    """w_ij = r_ij - t_ij. Positive: the two systems want to be apart. Negative: together."""
    return risk(i, j) - traffic(i, j)


def cost(zones):
    """Sum over pairs of t_ij [z_i != z_j] + r_ij [z_i == z_j]. `zones` has one zone id per system."""
    return sum(traffic(i, j) * (zones[i] != zones[j]) + risk(i, j) * (zones[i] == zones[j])
               for i, j in PAIRS)


def decode(index, K):
    """Zones of all 8 systems for a partition index, with the nurse in zone 0.
    Free system k is in zone (index // K**k) % K. For K=2 that is bit k, the order Qiskit uses for qubit k."""
    return (0,) + tuple((index // K ** k) % K for k in range(len(FREE)))


def cost_table(K):
    """Cost of every partition, listed by partition index (see `decode`)."""
    return [cost(decode(index, K)) for index in range(K ** len(FREE))]


def describe(index, K):
    zones = decode(index, K)
    return {chr(ord("A") + z): [SYSTEMS[i] for i in range(len(SYSTEMS)) if zones[i] == z] for z in range(K)}


def total_traffic():
    return sum(traffic(i, j) for i, j in PAIRS)


def total_risk():
    return sum(risk(i, j) for i, j in PAIRS)
