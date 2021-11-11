import unifloc.pipe._pipe as pip

class Annul(pip.Pipe):
    def __init__(
        self,
        fluid: fl.FluidFlow,
        ambient_temperature_distribution: amb.AmbientTemperatureDistribution,
        bottom_depth: float,
        d_casing: Union[float, pd.DataFrame, dict],
        d_tubing: Union[float, pd.DataFrame, dict],
        s_wall: float,
        roughness: float,
        trajectory: traj.Trajectory,
    ):
