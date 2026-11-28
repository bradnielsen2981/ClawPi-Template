# -*- coding: utf-8 -*-
# Settings and helpers shared by the routines.
import time

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
        None: The routine was stopped (e.g. Stop set robot.routine to "Ready"), so it should end.
    """
    start = time.time()
    interrupted_by = robot.routine
    if interrupted_by not in robot.behaviours:
        return None # not a behaviour: the routine was stopped
    resume_at = None
    while robot.routine == interrupted_by:
        if robot.movement_controlled:
            resume_at = None # the behaviour is still moving the robot
        elif resume_at is None:
            resume_at = time.time() + PAUSETIME
        elif time.time() >= resume_at:
            # Only resumes if nothing (e.g. Stop) changed robot.routine in the meantime - checked inside the lock
            if robot.resume_routine(routine_name, by=interrupted_by):
                return time.time() - start
        time.sleep(0.05)
    return None
