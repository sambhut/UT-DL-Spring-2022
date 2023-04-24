from . import custom_runner,utils


# TODO: Wrap TeamRunner and Team in ray if possible
try:
    import ray


    @ray.remote
    class RayMatch(custom_runner.Match):
        pass

    @ray.remote
    class RayTeamRunner(custom_runner.TeamRunner):
        pass

    @ray.remote
    class RayDataRecorder(utils.DataRecorder):
        pass

    @ray.remote
    class RayVideoRecorder(utils.VideoRecorder):
        pass

    RayMatchException = ray.exceptions.RayTaskError

    get = ray.get
    init = ray.init

except ImportError:
    ray = None
