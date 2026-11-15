"""Reference skin contacts and the native 15-segment to 10-slot reduction."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .layout import CHAINS


DEFAULT_CONTACT_DISTANCE_M = 0.005


def part_meshes(scene, surface_geoms):
    m = scene.model
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    meshes = []
    for part, body in zip(scene.layout['parts'], scene.part_ids):
        pieces = []
        for name in surface_geoms[part]:
            g = m.geom(name).id
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_bodyid[g] != body:
                raise ValueError('Expected a part-local visual mesh')
            mesh = m.geom_dataid[g]
            v0, nv = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            f0, nf = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
            world = m.mesh_vert[v0:v0+nv] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            local = (world-d.xpos[body]) @ d.xmat[body].reshape(3, 3)
            pieces.append(trimesh.Trimesh(local, m.mesh_face[f0:f0+nf], process=False))
        meshes.append(trimesh.util.concatenate(pieces))
    return meshes


def segment_labels(joints, vertices):
    segments = [(chain[i], chain[i+1]) for chain in CHAINS for i in range(3)]
    start = joints[[a for a, _ in segments]]
    axis = joints[[b for _, b in segments]] - start
    offset = vertices[:, None] - start
    along = np.clip((offset*axis).sum(-1) / np.maximum((axis*axis).sum(-1), 1e-12), 0, 1)
    distance = np.linalg.norm(offset - along[..., None]*axis, axis=-1)
    return distance.argmin(-1)


def reference_contacts(scene, task, threshold=DEFAULT_CONTACT_DISTANCE_M):
    layout, arrays = scene.layout, task.demo.arrays
    shape = (len(arrays['time_s']), len(layout['hands']), len(layout['parts']), 10)
    flags, points = np.zeros(shape, bool), np.zeros((*shape, 3), np.float32)
    meshes = part_meshes(scene, task.object_binding['surface_geoms'])
    for h, side in enumerate(layout['hands']):
        hand = task.demo.hands[side]
        for t, vertices in enumerate(hand.vertices_m):
            labels = segment_labels(hand.joints_m[t], vertices)
            for p, mesh in enumerate(meshes):
                rotation = Rotation.from_quat(arrays['part_quat_wxyz'][t, p], scalar_first=True).as_matrix()
                local = (vertices - arrays['part_pos_m'][t, p]) @ rotation
                # Outside the expanded AABB, no triangle can be within the contact threshold.
                # This also avoids enormous proximity searches for hands far from the object.
                near = ((local >= mesh.bounds[0]-threshold) & (local <= mesh.bounds[1]+threshold)).all(-1)
                closest, distance = np.zeros_like(local), np.full(len(local), np.inf)
                if near.any():
                    closest[near], distance[near], _ = trimesh.proximity.closest_point(mesh, local[near])
                segment_flags, segment_points = np.zeros(15, bool), np.zeros((15, 3))
                for s in range(15):
                    mask = (labels == s) & (distance <= threshold)
                    segment_flags[s] = mask.any()
                    if mask.any():
                        segment_points[s] = closest[mask].mean(0)
                for f in range(5):
                    for slot, ids in [(2*f, [3*f]), (2*f+1, [3*f+1, 3*f+2])]:
                        valid = np.asarray(ids)[segment_flags[ids]]
                        flags[t, h, p, slot] = len(valid) > 0
                        if len(valid):
                            points[t, h, p, slot] = segment_points[valid].mean(0)
            if (t+1) % 100 == 0:
                print(f'Contacts {side}: {t+1}/{len(arrays["time_s"])}', flush=True)
    return flags, points
