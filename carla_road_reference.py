"""Visible road reference from native tags, aligned depth, and a saved map footprint."""
import numpy as np


EDGE_DISAGREEMENT_POLICY = "At most 0.1% semantic/instance disagreement; preserve both native streams and ignore every disagreeing pixel"


def checked_appearance(tags, instance, config, row):
    """Validate the declared native streams; isolate mismatches for exclusion.

    Strict historical captures still reject every mismatch. Dataset-style
    captures may declare the existing 0.1% edge policy; none of those pixels is
    allowed to become a road/non-road target.
    """
    if instance.dtype != np.uint8 or instance.shape != (*tags.shape, 3):
        raise ValueError("Expected matching raw RGB instance evidence")
    ids = instance[...,1].astype(np.uint16) + 256*instance[...,2].astype(np.uint16)
    appearance = np.isin(ids, config["reference"]["appearance_actor_ids"])
    if int(appearance.sum()) != row["appearance_pixels"]:
        raise ValueError("Saved appearance-instance count differs")
    disagreement = instance[...,0] != tags
    count = int(disagreement.sum())
    if count != row.get("semantic_instance_disagreement_pixels", 0):
        raise ValueError("Saved semantic/instance mismatch count differs")
    if count and (config.get("dataset_edge_disagreement_policy") != EDGE_DISAGREEMENT_POLICY
                  or count > tags.size*.001):
        raise ValueError("Undeclared or excessive semantic/instance disagreement")
    appearance &= ~disagreement
    return appearance, disagreement


def build_reference(tags, depth, camera_to_world, config, footprint, appearance_mask=None):
    """Return 0/1/255 labels and diagnostics without reading any model prediction."""
    if tags.shape != depth.shape or tags.ndim != 2 or not np.isfinite(depth).all() or depth.min() < 0:
        raise ValueError("Expected aligned finite metric depth and raw class IDs")
    if tags.dtype != np.uint8 or tags.max() > 28 or config["schema_version"] not in (2, 3):
        raise ValueError("Unexpected label/depth reference protocol")
    if config["schema_version"] == 3:
        if (appearance_mask is None or appearance_mask.shape != tags.shape or appearance_mask.dtype != bool
                or not (tags[appearance_mask] == config["reference"]["appearance_native_class"]).all()):
            raise ValueError("Expected the owned appearance-instance mask with its exact native class")
    elif appearance_mask is not None:
        raise ValueError("Legacy captures do not contain appearance instances")
    height, width = tags.shape
    k = np.array(config["intrinsics_source"], dtype=np.float64)
    # CARLA depth is camera-forward depth; image y points down, Unreal z points up.
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    local = np.stack([depth, (u-k[0,2])*depth/k[0,0], -(v-k[1,2])*depth/k[1,1]], axis=0)
    transform = np.array(camera_to_world, dtype=np.float64)
    points = np.einsum("ij,jhw->ihw", transform[:3,:3], local) + transform[:3,3,None,None]
    parameters = config["reference"]
    grid = parameters["grid"]
    gx = np.rint((points[0]-grid["origin_xy_m"][0])/grid["cell_m"]).astype(np.int32)
    gy = np.rint((points[1]-grid["origin_xy_m"][1])/grid["cell_m"]).astype(np.int32)
    radius = round(parameters["boundary_uncertainty_m"] / grid["cell_m"])
    inside_grid = ((gx >= radius) & (gy >= radius) & (gx < footprint.shape[1]-radius)
                   & (gy < footprint.shape[0]-radius))
    clipped_x, clipped_y = gx.clip(radius, footprint.shape[1]-radius-1), gy.clip(radius, footprint.shape[0]-radius-1)
    votes = [footprint[clipped_y+dy, clipped_x+dx].astype(bool)
             for dx in (-radius, 0, radius) for dy in (-radius, 0, radius)]
    inside, outside = np.logical_and.reduce(votes), ~np.logical_or.reduce(votes)
    range_ok = (depth > .05) & (depth < parameters["geometry_range_m"])
    flat = abs(points[2]) <= parameters["surface_height_tolerance_m"]
    unknown = tags == 0
    if appearance_mask is not None:
        unknown |= appearance_mask
    eligible = unknown & inside_grid & range_ok
    result = np.where(np.isin(tags, [1,24]), 1, 0).astype(np.uint8)
    result[unknown] = 255
    result[eligible & inside & flat] = 1
    # Known visible classes, off-footprint ground and objects above the road are non-road.
    result[eligible & (outside | (abs(points[2]) > .20))] = 0
    central_road = unknown & inside & inside_grid & range_ok & (depth < 30)
    return result, {"native_road_pixels": int(np.isin(tags,[1,24]).sum()),
        "geometric_road_pixels": int((result[unknown] == 1).sum()),
        "ignored_pixels": int((result == 255).sum()),
        "ground_height_abs_median_m": float(np.median(abs(points[2][central_road]))) if central_road.any() else None,
        "near_map_road_flat_fraction": float(flat[central_road].mean()) if central_road.any() else None}
