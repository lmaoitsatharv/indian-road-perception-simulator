"""Shared RGB post-processing for the viewer and recorded model camera."""
def configure_rgb(blueprint):
    settings = {"motion_blur_intensity": "0", "bloom_intensity": "0", "lens_flare_intensity": "0"}
    for key, value in settings.items():
        if not blueprint.has_attribute(key):
            raise RuntimeError(f"Required RGB camera setting is unavailable: {key}")
        blueprint.set_attribute(key, value)
    return {key: float(blueprint.get_attribute(key).as_float()) for key in settings}
