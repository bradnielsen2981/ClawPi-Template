# Robot routines: each file holds one routine function that RobotInterface attaches as a method.
# Shared settings for the routines live here.
import time

# Speeds used by the movement routines
DRIVE_SPEED = 0.15 # m/s
TURN_SPEED = 0.5   # rad/s

# Seconds a routine waits after a behaviour (see robot_behaviours/) has finished interrupting it, before carrying on
PAUSETIME = 1.0


def pause_for_behaviour(robot, routine_name):
    """
    Call this from a routine's loop when robot.routine is no longer the routine's name.

    A behaviour that takes over (e.g. stay_within_quadrant at a boundary line) sets robot.routine to its own name.
    This waits until the behaviour has finished moving the robot, waits PAUSETIME more, then sets robot.routine
    back to routine_name so the routine can carry on. A routine that would rather stop when interrupted can
    simply end instead of calling this.

    Returns:
        float: Seconds paused, so the routine can add them to its time limit.
        None: The routine was stopped (e.g. stop_routine() set robot.routine to "Ready"), so it should end.
    """
    start = time.time()
    resume_at = None
    while robot.routine in robot.behaviours: # still interrupted by a behaviour
        if robot.movement_controlled:
            resume_at = None # the behaviour is still moving the robot
        elif resume_at is None:
            resume_at = time.time() + PAUSETIME
        elif time.time() >= resume_at:
            robot.routine = routine_name
            return time.time() - start
        time.sleep(0.05)
    return None
