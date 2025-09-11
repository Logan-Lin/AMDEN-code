import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra
from ase.neighborlist import NeighborList
from ase.data import covalent_radii
from ase.symbols import symbols2numbers



def check_ring_is_periodic(ring, offsets):
    total_offset = np.zeros(3)
    for i in range(len(ring) - 1):
        total_offset += offsets[ring[i], ring[i+1]]
    total_offset += offsets[ring[-1], ring[0]]
    return np.all(total_offset == 0)

def find_rings(ats, radii_factor=1.3, repeat=(1, 1, 1)):
    s = ats.repeat(repeat)
    pos = s.get_positions()
    nat = len(s)
    lat = s.get_cell()
    els = s.get_chemical_symbols()
    radii = covalent_radii[symbols2numbers(els)]
    
    nl = NeighborList(radii * radii_factor, self_interaction=False, bothways=False, skin=0.)
    nl.update(s)

    d = np.zeros((nat, nat))
    all_offsets = np.zeros((nat, nat, 3), dtype=int)



    for i in range(nat):
        indices, offsets = nl.get_neighbors(i)
        rs = pos[indices, :] + offsets @ lat - pos[i, :]
        ds = np.linalg.norm(rs, axis=1)
        d[i, indices] = ds
        d[indices, i] = ds
        all_offsets[i, indices] = offsets
        all_offsets[indices, i] = -offsets

    d = csr_array(d)


    
    rings = {}
    for i in range(len(ats)):
        indices, offsets = nl.get_neighbors(i)
        for j, _ in zip(indices, offsets):
            d_tmp = d.copy()
            d_tmp[i, j] = 0
            d_tmp[j, i] = 0
            d_tmp.eliminate_zeros()
            dist_matrix, predecessors = dijkstra(d_tmp, indices=i, return_predecessors=True, directed=False, unweighted=True, limit=np.inf)
            if dist_matrix[j] < np.inf:
                k = j
                ring = [k]
                while predecessors[k] != i:
                    k = predecessors[k]
                    ring.append(k)
                ring.append(i)

                if not check_ring_is_periodic(ring, all_offsets):
                    print('WARNING: ring is wrapping around periodic cell!')
                    continue

                ring = [x % len(ats) for x in ring]  # take it back to primary cell
                rings[tuple(sorted(ring))] = ring
            else:
                print('WARNING: no path found between atoms', i, j)


    return list(rings.values())
