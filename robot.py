# This is where your main robot code resides. It brings everything together:
#   RobotInterface (interfaces/robotinterface.py) - the robot's hardware, camera, sensors, sound and safety
#   Routines (interfaces/robot_routines/)          - routines: loops that do one thing, e.g. drive toward a colour
#   Robot (this file)                              - tasks: a combination of routines that achieves a goal
#
# The self.task and self.routine variables keep track of what the robot is doing:
#   self.task is the task running (e.g. 'automated_search') or None,
#   self.routine is the routine running inside it (e.g. 'face_tracking') or "Ready".
# Several threads use them at once (Flask requests, the voice agent, behaviours), so READ them freely but CHANGE them
# only with self.set_task(), self.stop_task(), self.set_routine() and self.stop_routine(), which use a lock.

import logging, sys, os, time
from interfaces.robotinterface import RobotInterface
from interfaces.robot_routines import Routines
from interfaces.databaseinterface import Database

class Robot(RobotInterface, Routines):
    # The tasks below, by name: used by ROBOT.start_task(name) and the voice agent ("start automated search")
    TASKS = ['automated_search']

    def __init__(self, DATABASE=None):
        # Initialize the underlying hardware and camera threads
        super().__init__()

        # Connect the database securely
        if DATABASE is None:
            self.DATABASE = Database("databases/test.db")
        else:
            self.DATABASE = DATABASE
        return

    # Write a function for automated search
    def automated_search(self, timelimit=300):
        """
        Searches the area on its own until stopped or the time limit is reached.
        Students can write their autonomous logic here!
        Use self.move_direction_until_detection(), self.move_toward_colour_detected(), etc.

        Args:
            timelimit (int/float): Seconds to search for. Defaults to 300.
        """
        self.set_task('automated_search')
        endtime = time.time() + timelimit
        try:
            while self.task == 'automated_search' and time.time() < endtime:

                # STUDENT CODE GOES HERE

                time.sleep(0.05) # stops an empty loop using all of the CPU
        except Exception as e:
            self.emergency_shutdown(f"automated_search crashed: {e}")
            raise
        finally:
            # Always runs - finished, stopped or crashed
            if self.task == 'automated_search':
                self.stop_task()
        return

# Only execute if this is the main file, good for testing code
if __name__ == '__main__':
    print("\033c")
    ROBOT = Robot(DATABASE=None)

    try:
        ROBOT.stop()
        input("Press enter to begin testing:")

        # You can test student functions here
        # ROBOT.automated_search(timelimit=10)
        # data = ROBOT.move_direction_until_detection(movetype='forward', distanceto=50, detection_types=['lidar','colour'],
        #                                             confirmlevel=2, detection_colours=['red'], timelimit=10)
        # print(data)

    except KeyboardInterrupt:
        print("\n[!] Testing interrupted.")
    finally:
        # Guarantee safe shutdown
        ROBOT.shutdown()
