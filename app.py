# Use Control + K and Control + 0 to minimise function names

from flask import *
from interfaces.databaseinterface import Database
from interfaces.hashinginterface import *
from robot import Robot
from interfaces.loggerinterface import setup_logging
import logging, time, sys, os, signal, threading, uuid

#---SETUP LOGGING------------------------------------------------------
# Each interface logs to its own file in the logs folder (see interfaces/loggerinterface.py):
# flask.log, robot.log, camera.log, map.log, database.log ... Each file is emptied when the app starts.
LOGGING = True              # False turns off ALL logging (every log file and the terminal warnings)
LOG_LEVEL = logging.INFO    # logging.DEBUG also logs every SQL query; logging.WARNING only problems
setup_logging(enabled=LOGGING, level=LOG_LEVEL)

#---CONFIGURE APP---------------------------------------------------
app = Flask(__name__)
sys.tracebacklimit = 10
app.config['SECRET_KEY'] = "ClawPi2026"

#---EMBED OBJECTS---------------------------------------------------
DATABASE = Database("databases/test.db") # logs to logs/database.log
ROBOT = None

#---ONE USER AT A TIME-----------------------------------------------
# Only one browser can be logged in. The logged in browser holds ACTIVE_USER['token'].
# If it goes quiet for LOGIN_TIMEOUT seconds (tab closed without logging out) the lock is released.
LOGIN_TIMEOUT = 60
ACTIVE_USER = {'token': None, 'last_seen': 0}
ACTIVE_USER_LOCK = threading.Lock()
PUBLIC_ROUTES = ('login', 'logout', 'static')

def someone_else_logged_in():
    return ACTIVE_USER['token'] is not None and time.time() - ACTIVE_USER['last_seen'] < LOGIN_TIMEOUT

# Runs before every request: only the logged in browser can use the app
@app.before_request
def check_active_user():
    if request.endpoint in PUBLIC_ROUTES:
        return
    with ACTIVE_USER_LOCK:
        is_active = 'token' in session and session['token'] == ACTIVE_USER['token'] and someone_else_logged_in()
        if is_active:
            ACTIVE_USER['last_seen'] = time.time()
    if not is_active:
        session.clear()
        if request.method == 'POST': # dashboard buttons
            return jsonify({'message':'not logged in'}), 401
        return redirect('/') # pages

#---VIEW FUNCTIONS----------------------------------------------------
# Backdoor
@app.route('/backdoor')
def backdoor():
    if session.get('permission') != 'admin': # admin users only
        app.logger.warning(f"Backdoor refused for {request.remote_addr}: not an admin")
        return jsonify({'message':'admin permission required'}), 403
    results = DATABASE.ViewQuery("SELECT * FROM users") if DATABASE else []
    return jsonify(results)

# Login as the admin user
@app.route('/', methods=["GET","POST"])
def login():
    if 'token' in session and session['token'] == ACTIVE_USER['token']:
        return redirect('/dashboard')
    app.logger.info("Login")
    message = None
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        if email == 'admin@admin' and password == 'admin':
            with ACTIVE_USER_LOCK:
                if someone_else_logged_in():
                    app.logger.warning(f"Login refused for {email} from {request.remote_addr}: another user is logged in")
                    message = "Another user is already logged in to the robot. Try again when they log out."
                else:
                    session.clear()
                    session['userid'] = 1
                    session['permission'] = 'admin'
                    session['token'] = uuid.uuid4().hex
                    ACTIVE_USER['token'] = session['token']
                    ACTIVE_USER['last_seen'] = time.time()
                    app.logger.info(f"Login: {email} from {request.remote_addr}")
                    return redirect('./dashboard') #takes 3 seconds for the CAMERA to start
        else:
            app.logger.warning(f"Failed login for {email} from {request.remote_addr}")
            message = "Incorrect email or password."
    return render_template("login.html", data=message)

# Dashboard for the robot
@app.route('/dashboard', methods=['GET','POST'])
def dashboard():
    if not 'userid' in session:
        return redirect('/')
    loaded = 0
    if ROBOT:
        loaded = 1
    return render_template('dashboard.html', robot_loaded=loaded)

# load the robot
@app.route('/load_robot', methods=['GET','POST'])
def load_robot():
    global ROBOT
    if not ROBOT:
        app.logger.info('Loading Robot')
        ROBOT = Robot(DATABASE)
        time.sleep(3) #takes 3 seconds to load the robot
        #ROBOT.SOUND.say("Hello my name is Wally")
    return jsonify({'message':'robot loaded'})

# YOUR FLASK CODE------------------------------------------------------------------------

# Look Down
@app.route('/look_down', methods=['GET','POST'])
def look_down():
    app.logger.info('Looking Down')
    if ROBOT:
        pass #ROBOT.SOUND.say("Looking Down")
    return jsonify({'message':'look down'})

# Stop
@app.route('/stop', methods=['GET','POST'])
def stop():
    app.logger.info('stopping robot')
    if ROBOT:
        ROBOT.stop_behaviour() # also ends any held dashboard button
    return jsonify({'message':'stop'})

# DASHBOARD HOLD-DOWN BUTTONS (movement, claw, camera pan/tilt)----------------------------
# The dashboard sends /hold every 0.5 s while a button is held and /release when it is let go.
# If the robot hears nothing for 3 seconds it stops by itself (see HOLD_TIMEOUT in robotinterface.py).
# Each request carries a rising 'seq' number. Requests can arrive out of order, so an older one is ignored:
# a late /hold must never restart the robot after the button was released.
HOLD_SEQ = {'last': 0}
HOLD_SEQ_LOCK = threading.Lock()

def is_newest_hold_request():
    try:
        seq = int(request.form.get('seq', 0))
    except ValueError:
        return False
    with HOLD_SEQ_LOCK:
        if seq and seq <= HOLD_SEQ['last']:
            return False
        HOLD_SEQ['last'] = max(seq, HOLD_SEQ['last'])
        return True

@app.route('/hold', methods=['POST'])
def hold():
    if not ROBOT:
        return jsonify({'message':'robot not loaded'})
    if not is_newest_hold_request():
        return jsonify({'message':'old request ignored'})
    action = request.form.get('action', '')
    if not ROBOT.hold_action(action):
        return jsonify({'message':'unknown action'}), 400
    pan, tilt = ROBOT.get_camera_angles()
    return jsonify({'message':'holding', 'action':action, 'pan':pan, 'tilt':tilt})

@app.route('/release', methods=['POST'])
def release():
    if not ROBOT:
        return jsonify({'message':'robot not loaded'})
    if not is_newest_hold_request():
        return jsonify({'message':'old request ignored'})
    ROBOT.release_hold()
    pan, tilt = ROBOT.get_camera_angles()
    return jsonify({'message':'released', 'pan':pan, 'tilt':tilt})

# SENSORS CARD---------------------------------------------------------------------------
# Battery voltage, IMU and the 4 channel line follower under the robot (polled by the dashboard)
@app.route('/sensor_data', methods=['GET'])
def sensor_data():
    robot = ROBOT
    if not robot:
        return jsonify({'loaded':False})
    imu = robot.get_imu_sensor(timeout=0) # timeout=0: use the latest reading, don't wait
    return jsonify({'loaded':True,
                    'voltage':robot.get_battery_voltage(timeout=0),
                    'imu':imu and {'rpy':imu['rpy'], 'accel':imu['accel'], 'gyro':imu['gyro']},
                    'line':robot.LINE_FOLLOWER.get_line_data()}) # sensors, on_line, position, junction

# SLAM MAP CARD---------------------------------------------------------------------------
# A live picture of the lidar SLAM map (see interfaces/mappinginterface.py), streamed like the camera
MAP_FPS = 2 # map pictures per second sent to the dashboard

def mapstream(robot):
    """Map streaming generator. Ends when the robot is shut down."""
    mapping = robot.MAPPING
    while ROBOT is robot and mapping.status != "Stop":
        frame = mapping.get_jpeg_map_image(quality=70, size=480)
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1 / MAP_FPS)

@app.route('/mapfeed')
def mapfeed():
    """Map streaming route. Put this in the src attribute of an img tag."""
    robot = ROBOT
    if not robot:
        return '', 204
    response = Response(mapstream(robot), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-store'
    return response

# Is the SLAM program running? (polled by the dashboard)
@app.route('/map_status', methods=['GET'])
def map_status():
    robot = ROBOT
    if not robot:
        return jsonify({'loaded':False})
    return jsonify({'loaded':True, 'status':robot.MAPPING.status}) # Init, Ready, Mapping, Fail or Stop

# Start SLAM. Waits (up to 30 s) for the first map, so the dashboard shows a spinner meanwhile.
@app.route('/start_mapping', methods=['POST'])
def start_mapping():
    robot = ROBOT
    if not robot:
        return jsonify({'message':'robot not loaded'})
    app.logger.info('start mapping')
    started = robot.MAPPING.start_mapping()
    return jsonify({'message':'mapping' if started else 'mapping failed - see logs/map.log',
                    'status':robot.MAPPING.status})

# Stop SLAM. The last map stays on screen.
@app.route('/stop_mapping', methods=['POST'])
def stop_mapping():
    robot = ROBOT
    if not robot:
        return jsonify({'message':'robot not loaded'})
    app.logger.info('stop mapping')
    robot.MAPPING.stop_mapping()
    return jsonify({'message':'mapping stopped', 'status':robot.MAPPING.status})


# DASHBOARD CAMERA CONTROLS---------------------------------------------------------------

# Models the dashboard's Detect Object menu can load (see interfaces/detection_tasks/detect_object.py)
OBJECT_MODELS = ['yolo26n', 'best_traffic_26', 'best_garbage_26', 'best_FourLineFollowing']

# Point the camera straight ahead (pan and tilt are hold-down buttons, see /hold)
@app.route('/camera_centre', methods=['POST'])
def camera_centre():
    if not ROBOT:
        return jsonify({'message':'robot not loaded'})
    pan, tilt = ROBOT.centre_camera_view()
    return jsonify({'message':'camera centred', 'pan':pan, 'tilt':tilt})

# Turn one detection task on or off. Form fields:
#   task    - colour, line, object or pose
#   on      - 1 to turn on (or update the options), 0 to turn off
#   options - comma separated: colours (red,green), line parts (line,crossing),
#             a model name (yolo26n) or gestures (left_arm_up,fallen)
@app.route('/set_detection', methods=['POST'])
def set_detection():
    if not ROBOT:
        return jsonify({'message':'robot not loaded'})
    camera = ROBOT.CAMERA
    task = request.form.get('task', '')
    on = request.form.get('on') == '1'
    options = [o for o in request.form.get('options', '').split(',') if o]
    names = {'colour':'detect_colour', 'line':'detect_line', 'object':'detect_object', 'pose':'detect_pose'}
    if task not in names:
        return jsonify({'message':'unknown task'}), 400

    if on and task == 'colour':
        options = [c for c in options if c in camera.lab_colours]
        camera.set_detection_colours(options)
    elif on and task == 'line':
        camera.set_line_detection_options(lines='line' in options, crossings='crossing' in options)
    elif on and task == 'object':
        model = options[0] if options and options[0] in OBJECT_MODELS else 'yolo26n'
        if not camera.load_detection_model(model): # takes a few seconds the first time
            return jsonify({'message':f'could not load {model}'}), 500
    elif on and task == 'pose':
        camera.set_pose_gestures(options)

    if on and (options or task == 'pose'): # pose with no gestures still finds people
        camera.add_detection_task(names[task])
    else:
        camera.remove_detection_task(names[task])
    app.logger.info(f'detection {task} {"on" if on else "off"} {options}')
    return jsonify({'message':'detection updated', 'tasks':camera.detection_tasks})

# The latest camera detection data, frame rate and active tasks (polled by the dashboard)
@app.route('/detection_data', methods=['GET'])
def detection_data():
    if not ROBOT:
        return jsonify({'loaded':False})
    camera = ROBOT.CAMERA
    return jsonify({'loaded':True, 'fps':camera.framerate, 'tasks':list(camera.detection_tasks),
                    'data':camera.get_detection_data()})

# CAMERA CODE-(do not touch this!!)-------------------------------------------------------

# Continually gets the frame from the pi camera
def videostream(camera):
    """Video streaming generator function. Ends the stream when the camera stops."""
    # camera is kept locally so shutdown_robot() setting ROBOT = None can't crash the stream
    while camera.status == "Running":
        frame = camera.get_jpeg_frame() #can change quality to test efficiency
        if not frame:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Embeds the videofeed by returning a continual stream as above
@app.route('/videofeed')
def videofeed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    robot = ROBOT
    if not robot:
        return '', 204
    response = Response(videostream(robot.CAMERA), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-store' # a live stream: never keep a copy
    return response

# Turn on detection mode
@app.route('/turn_on_detection', methods=['GET','POST'])
def turn_on_detection():
    app.logger.info('turn on detection')
    if ROBOT:
        ROBOT.CAMERA.detect_all()
    return jsonify({'message':'Detection mode on!!'})

# Turn off detection mode
@app.route('/turn_off_detection', methods=['GET','POST'])
def turn_off_detection():
    app.logger.info('turn off detection')
    if ROBOT:
        ROBOT.CAMERA.end_detection()
    return jsonify({'message':'Detection mode off!!'})

# Log out
@app.route('/logout')
def logout():
    app.logger.info('logging off')
    with ACTIVE_USER_LOCK:
        if session.get('token') and session.get('token') == ACTIVE_USER['token']:
            ACTIVE_USER['token'] = None # free the robot for the next user
    session.clear()
    return redirect('/')

# Shut down the robot
@app.route('/shutdown_robot', methods=['GET','POST'])
def shutdown_robot():
    app.logger.info("Shutting down robot")
    global ROBOT
    if ROBOT:
        ROBOT.shutdown() # stops motors, mapping, lidar and camera, then releases the ROS connection
        time.sleep(0.5)
        #ROBOT.SOUND.say("Goodbye")
        ROBOT = None
    return jsonify({'message':'Shutting Down'})

# Keeps the login alive while the dashboard is open (the check is done in check_active_user)
@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    return jsonify({'message':'ok'})

# Exit the web server
@app.route('/exit', methods=['GET','POST'])
def exit_app():
    app.logger.info("Exiting")
    shutdown_robot()
    # Safely sends an interrupt signal to the current process
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({'message':'Exiting'})

#---------------------------------------------------------------------------
# main method called web server application
if __name__ == '__main__':
    # CRITICAL: use_reloader=False prevents Flask from spawning a second process 
    # which crashes the robot's hardware GPIO and Camera connections.
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)