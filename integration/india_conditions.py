"""Named, logged appearance conditions; not a tyre-friction or traffic calibration."""
PRESETS = {
    "clear": dict(sun_altitude_angle=55, sun_azimuth_angle=135, cloudiness=15),
    "overcast": dict(sun_altitude_angle=45, sun_azimuth_angle=135, cloudiness=85),
    "wet": dict(sun_altitude_angle=35, sun_azimuth_angle=135, cloudiness=75,
                precipitation=25, precipitation_deposits=30, wetness=70),
    "dusk": dict(sun_altitude_angle=8, sun_azimuth_angle=135, cloudiness=30),
}


def weather_parameters(name, physical_atmosphere=False):
    import carla
    if physical_atmosphere:
        # Match the nonzero atmosphere in CARLA's named noon/sunset presets.
        # Constructor defaults for scattering/Mie/fog are zero, which produced
        # black daytime skies in the custom world. Keep historical calls intact.
        values=dict(PRESETS[name])
        if name in ("overcast","wet"):
            # Native CloudyNoon uses 60. A seven-condition native probe showed
            # 85 produces black clouds in this custom world's low-quality sky.
            values["cloudiness"]=60
        return carla.WeatherParameters(wind_intensity=10, fog_density=2 if name=="clear" else 3,
            fog_distance=.75, fog_falloff=.1, scattering_intensity=1.,
            mie_scattering_scale=.03, rayleigh_scattering_scale=.0331, **values)
    return carla.WeatherParameters(**PRESETS[name])
