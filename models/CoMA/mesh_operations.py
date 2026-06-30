import math
import heapq
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
import scipy.sparse as sp


class _ChumpyStub:
    """Allows trusted MANO pickles to load when chumpy is unavailable."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


class _MANOUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('chumpy'):
            return _ChumpyStub
        return super().find_class(module, name)


@dataclass
class SimpleMesh:
    """Minimal mesh container used by the CoMA preprocessing code."""

    v: np.ndarray
    f: np.ndarray

    def __post_init__(self):
        self.v = np.asarray(self.v, dtype=np.float64)
        self.f = np.asarray(self.f, dtype=np.int64)

        if self.v.ndim != 2 or self.v.shape[1] != 3:
            raise ValueError(f'Invalid mesh vertex shape: {self.v.shape}')
        if self.f.ndim != 2 or self.f.shape[1] != 3:
            raise ValueError(f'Invalid mesh face shape: {self.f.shape}')

def row(A):
    return A.reshape((1, -1))

def col(A):
    return A.reshape((-1, 1))

def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

    vpv = sp.csc_matrix((len(mesh_v),len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:,i]
        JS = mesh_f[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv

def get_vertices_per_edge(mesh_v, mesh_f):
    """Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()"""

    vc = sp.coo_matrix(get_vert_connectivity(mesh_v, mesh_f))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:,0] < result[:,1]] # for uniqueness

    return result


def vertex_quadrics(mesh):
    """Computes a quadric for each vertex in the Mesh.

    Returns:
       v_quadrics: an (N x 4 x 4) array, where N is # vertices.
    """

    # Allocate quadrics
    v_quadrics = np.zeros((len(mesh.v), 4, 4,))

    # For each face...
    for f_idx in range(len(mesh.f)):

        # Compute normalized plane equation for that face
        vert_idxs = mesh.f[f_idx]
        verts = np.hstack((mesh.v[vert_idxs], np.array([1, 1, 1]).reshape(-1, 1)))
        u, s, v = np.linalg.svd(verts)
        eq = v[-1, :].reshape(-1, 1)
        eq = eq / (np.linalg.norm(eq[0:3]))

        # Add the outer product of the plane equation to the
        # quadrics of the vertices for this face
        for k in range(3):
            v_quadrics[mesh.f[f_idx, k], :, :] += np.outer(eq, eq)

    return v_quadrics

def _get_sparse_transform(faces, num_original_verts):
    verts_left = np.unique(faces.flatten())
    IS = np.arange(len(verts_left))
    JS = verts_left
    data = np.ones(len(JS))

    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij), shape=(len(verts_left) , num_original_verts ))

    return (new_faces, mtx)

def qslim_decimator_transformer(mesh, factor=None, n_verts_desired=None):
    """Return a simplified version of this mesh.

    A Qslim-style approach is used here.

    :param factor: fraction of the original vertices to retain
    :param n_verts_desired: number of the original vertices to retain
    :returns: new_faces: An Fx3 array of faces, mtx: Transformation matrix
    """

    if factor is None and n_verts_desired is None:
        raise Exception('Need either factor or n_verts_desired.')

    if n_verts_desired is None:
        n_verts_desired = math.ceil(len(mesh.v) * factor)

    Qv = vertex_quadrics(mesh)

    # fill out a sparse matrix indicating vertex-vertex adjacency
    vert_adj = get_vertices_per_edge(mesh.v, mesh.f)
    # vert_adj = sp.lil_matrix((len(mesh.v), len(mesh.v)))
    # for f_idx in range(len(mesh.f)):
    #     vert_adj[mesh.f[f_idx], mesh.f[f_idx]] = 1

    vert_adj = sp.csc_matrix((vert_adj[:, 0] * 0 + 1, (vert_adj[:, 0], vert_adj[:, 1])), shape=(len(mesh.v), len(mesh.v)))
    vert_adj = vert_adj + vert_adj.T
    vert_adj = vert_adj.tocoo()

    def collapse_cost(Qv, r, c, v):
        Qsum = Qv[r, :, :] + Qv[c, :, :]
        p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
        p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

        destroy_c_cost = p1.T.dot(Qsum).dot(p1)
        destroy_r_cost = p2.T.dot(Qsum).dot(p2)
        result = {
            'destroy_c_cost': destroy_c_cost,
            'destroy_r_cost': destroy_r_cost,
            'collapse_cost': min([destroy_c_cost, destroy_r_cost]),
            'Qsum': Qsum}
        return result

    # construct a queue of edges with costs
    queue = []
    for k in range(vert_adj.nnz):
        r = vert_adj.row[k]
        c = vert_adj.col[k]

        if r > c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)['collapse_cost']
        heapq.heappush(queue, (cost, (r, c)))

    # decimate
    collapse_list = []
    nverts_total = len(mesh.v)
    faces = mesh.f.copy()
    while nverts_total > n_verts_desired:
        e = heapq.heappop(queue)
        r = e[1][0]
        c = e[1][1]
        if r == c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)
        if cost['collapse_cost'] > e[0]:
            heapq.heappush(queue, (cost['collapse_cost'], e[1]))
            # print 'found outdated cost, %.2f < %.2f' % (e[0], cost['collapse_cost'])
            continue
        else:

            # update old vert idxs to new one,
            # in queue and in face list
            if cost['destroy_c_cost'] < cost['destroy_r_cost']:
                to_destroy = c
                to_keep = r
            else:
                to_destroy = r
                to_keep = c

            collapse_list.append([to_keep, to_destroy])

            # in our face array, replace "to_destroy" vertidx with "to_keep" vertidx
            np.place(faces, faces == to_destroy, to_keep)

            # same for queue
            which1 = [idx for idx in range(len(queue)) if queue[idx][1][0] == to_destroy]
            which2 = [idx for idx in range(len(queue)) if queue[idx][1][1] == to_destroy]
            for k in which1:
                queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
            for k in which2:
                queue[k] = (queue[k][0], (queue[k][1][0], to_keep))

            Qv[r, :, :] = cost['Qsum']
            Qv[c, :, :] = cost['Qsum']

            a = faces[:, 0] == faces[:, 1]
            b = faces[:, 1] == faces[:, 2]
            c = faces[:, 2] == faces[:, 0]

            # remove degenerate faces
            def logical_or3(x, y, z):
                return np.logical_or(x, np.logical_or(y, z))

            faces_to_keep = np.logical_not(logical_or3(a, b, c))
            faces = faces[faces_to_keep, :].copy()

        nverts_total = (len(np.unique(faces.flatten())))

    new_faces, mtx = _get_sparse_transform(faces, len(mesh.v))
    return new_faces, mtx


def _closest_face_barycentric(points, vertices, faces, chunk_size=64):
    """Find exact closest triangles and barycentric weights with NumPy."""
    triangles = vertices[faces]
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a

    normal = np.cross(ab, ac)
    normal_sq = np.einsum('fi,fi->f', normal, normal)
    safe_normal_sq = np.where(normal_sq > 0.0, normal_sq, 1.0)

    d00 = np.einsum('fi,fi->f', ab, ab)
    d01 = np.einsum('fi,fi->f', ab, ac)
    d11 = np.einsum('fi,fi->f', ac, ac)
    bary_denom = d00 * d11 - d01 * d01
    safe_bary_denom = np.where(np.abs(bary_denom) > 0.0, bary_denom, 1.0)

    closest_faces = np.empty(points.shape[0], dtype=np.int64)
    closest_weights = np.empty((points.shape[0], 3), dtype=np.float64)
    tolerance = 1e-12

    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        point_chunk = points[start:stop, None, :]
        chunk_size_actual = stop - start
        num_faces = faces.shape[0]

        best_distance = np.full(
            (chunk_size_actual, num_faces),
            np.inf,
            dtype=np.float64,
        )
        best_weights = np.zeros(
            (chunk_size_actual, num_faces, 3),
            dtype=np.float64,
        )

        # Candidate 1: orthogonal projection onto the triangle interior.
        ap = point_chunk - a[None, :, :]
        plane_scale = (
            np.einsum('mfi,fi->mf', ap, normal) / safe_normal_sq[None, :]
        )
        projected = point_chunk - plane_scale[:, :, None] * normal[None, :, :]
        projected_from_a = projected - a[None, :, :]
        d20 = np.einsum('mfi,fi->mf', projected_from_a, ab)
        d21 = np.einsum('mfi,fi->mf', projected_from_a, ac)
        weight_b = (
            d11[None, :] * d20 - d01[None, :] * d21
        ) / safe_bary_denom[None, :]
        weight_c = (
            d00[None, :] * d21 - d01[None, :] * d20
        ) / safe_bary_denom[None, :]
        weight_a = 1.0 - weight_b - weight_c
        plane_weights = np.stack((weight_a, weight_b, weight_c), axis=-1)
        plane_distance = np.sum((point_chunk - projected) ** 2, axis=-1)
        inside = (
            (normal_sq[None, :] > tolerance)
            & (np.abs(bary_denom)[None, :] > tolerance)
            & np.all(plane_weights >= -tolerance, axis=-1)
        )
        plane_distance = np.where(inside, plane_distance, np.inf)
        update = plane_distance < best_distance
        best_distance = np.where(update, plane_distance, best_distance)
        best_weights = np.where(
            update[:, :, None],
            plane_weights,
            best_weights,
        )

        # Candidates 2-4: closest points on each triangle edge. These also
        # cover vertex regions when the segment parameter is clamped.
        edge_specs = (
            (a, b, 0, 1),
            (b, c, 1, 2),
            (c, a, 2, 0),
        )
        for edge_start, edge_end, first_index, second_index in edge_specs:
            edge = edge_end - edge_start
            edge_length_sq = np.einsum('fi,fi->f', edge, edge)
            safe_edge_length_sq = np.where(
                edge_length_sq > 0.0,
                edge_length_sq,
                1.0,
            )
            edge_parameter = np.einsum(
                'mfi,fi->mf',
                point_chunk - edge_start[None, :, :],
                edge,
            ) / safe_edge_length_sq[None, :]
            edge_parameter = np.clip(edge_parameter, 0.0, 1.0)
            edge_point = (
                edge_start[None, :, :]
                + edge_parameter[:, :, None] * edge[None, :, :]
            )
            edge_distance = np.sum((point_chunk - edge_point) ** 2, axis=-1)
            edge_weights = np.zeros_like(best_weights)
            edge_weights[:, :, first_index] = 1.0 - edge_parameter
            edge_weights[:, :, second_index] = edge_parameter

            update = edge_distance < best_distance
            best_distance = np.where(update, edge_distance, best_distance)
            best_weights = np.where(
                update[:, :, None],
                edge_weights,
                best_weights,
            )

        local_rows = np.arange(chunk_size_actual)
        face_ids = np.argmin(best_distance, axis=1)
        closest_faces[start:stop] = face_ids
        closest_weights[start:stop] = best_weights[local_rows, face_ids]

    return closest_faces, closest_weights


def setup_deformation_transfer(source, target, use_normals=False):
    if use_normals:
        raise NotImplementedError('Normal transfer is not implemented.')

    nearest_faces, coefficients = _closest_face_barycentric(
        target.v,
        source.v,
        source.f,
    )
    rows = np.repeat(np.arange(target.v.shape[0]), 3)
    cols = source.f[nearest_faces].reshape(-1)
    values = coefficients.reshape(-1)

    matrix = sp.csc_matrix(
        (values, (rows, cols)),
        shape=(target.v.shape[0], source.v.shape[0]),
    )
    matrix.eliminate_zeros()
    return matrix


def generate_transform_matrices(mesh, factors):
    """Generates len(factors) meshes, each of them is scaled by factors[i] and
       computes the transformations between them.

    Returns:
       M: a set of meshes downsampled from mesh by a factor specified in factors.
       A: Adjacency matrix for each of the meshes
       D: Downsampling transforms between each of the meshes
       U: Upsampling transforms between each of the meshes
    """

    factors = map(lambda x: 1.0 / x, factors)
    M, A, D, U = [], [], [], []
    A.append(get_vert_connectivity(mesh.v, mesh.f).tocoo())
    M.append(mesh)

    for i,factor in enumerate(factors):
        ds_f, ds_D = qslim_decimator_transformer(M[-1], factor=factor)
        D.append(ds_D.tocoo())
        new_mesh_v = ds_D.dot(M[-1].v)
        new_mesh = SimpleMesh(v=new_mesh_v, f=ds_f)
        M.append(new_mesh)
        A.append(get_vert_connectivity(new_mesh.v, new_mesh.f).tocoo())
        U.append(setup_deformation_transfer(M[-1], M[-2]).tocoo())

    return M, A, D, U


def load_mano_template(model_path):
    """Load the template mesh stored in a MANO model pickle."""
    model_path = Path(model_path).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f'MANO model not found: {model_path}')

    # MANO model files use Python 2 string encoding. Only load trusted model
    # files because pickle is not safe for untrusted input.
    with model_path.open('rb') as model_file:
        model_data = _MANOUnpickler(
            model_file,
            encoding='latin1',
        ).load()

    try:
        vertices_data = model_data['v_template']
        faces_data = model_data['f']
    except KeyError as error:
        raise ValueError(
            f'{model_path} is missing the MANO field {error.args[0]!r}'
        ) from error

    # Some MANO releases store these values as chumpy arrays.
    vertices_data = getattr(vertices_data, 'r', vertices_data)
    faces_data = getattr(faces_data, 'r', faces_data)
    vertices = np.asarray(vertices_data, dtype=np.float64)
    faces = np.asarray(faces_data, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f'Invalid MANO template vertex shape: {vertices.shape}')
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f'Invalid MANO face shape: {faces.shape}')

    return SimpleMesh(v=vertices, f=faces)


def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor