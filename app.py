# Use Control + K and Control + 0 to minimise function names

from flask import *
from interfaces.databaseinterface import Database
from interfaces.hashinginterface import *
from robot import Robot
import logging, time, sys, os, signal

#---SETUP LOGGING------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "flask.log"),
    filemode="w",  # Clear the log each time the app starts
    level=logging.INFO
)

#---CONFIGURE APP---------------------------------------------------
app = Flask(__name__)
sys.tracebacklimit = 10
app.config['SECRET_KEY'] = "Type in a secret line of text"

#---EMBED OBJECTS---------------------------------------------------
DATABASE = Database("databases/test.db", app.logger)
ROBOT = None

#---VIEW FUNCTIONS----------------------------------------------------
# Backdoor
@app.route('/backdoor')
def backdoor():
    if DATABASE:
        results = DATABASE.ViewQuery("SELECT * FROM users")
    return jsonify(results)

# Login as the admin user
@app.route('/', methods=["GET","POST"])
def login():
    if 'userid' in session:
        return redirect('/dashboard')
    app.logger.info("Login")
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        if email == 'admin@admin' and password == 'admin': 
            session['userid'] = 1
            session['permission'] = 'admin'
            return redirect('./dashboard') #takes 3 seconds for the CAMERA to start
    return render_template("login.html")

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
        ROBOT.SOUND.say("Hello my name is Wally")
    return jsonify({'message':'robot loaded'})

# YOUR FLASK CODE------------------------------------------------------------------------

# Look Down
@app.route('/look_down', methods=['GET','POST'])
def look_down():
    app.logger.info('Looking Down')
    if ROBOT:
        ROBOT.SOUND.say("Looking Down")
    return jsonify({'message':'look down'})

# Stop
@app.route('/stop', methods=['GET','POST'])
def stop():
    app.logger.info('stopping robot')
    if ROBOT:
        ROBOT.stop_command()
        ROBOT.stop()
    return jsonify({'message':'stop'})


# DASHBOARD CAMERA CONTROLS---------------------------------------------------------------

# Models the dashboard's Detect Object menu can load (see interfaces/detection_tasks/detect_object.py)
OBJECT_MODELS = ['yolo26n', 'best_traffic_26', 'best_garbage_26', 'best_FourLineFollowing']

# Pan or tilt the camera a step: direction is up, down, left, right or centre
@app.route('/camera_move', methods=['POST'])
def camera_move():
    direction = request.form.get('direction', '')
    if not ROBOT:
        return jsonify({'message':'robot not loaded'})
    if direction not in ('up', 'down', 'left', 'right', 'centre'):
        return jsonify({'message':'unknown direction'}), 400
    pan, tilt = ROBOT.nudge_camera(direction)
    return jsonify({'message':'camera moved', 'pan':pan, 'tilt':tilt})

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
    return Response(videostream(robot.CAMERA), mimetype='multipart/x-mixed-replace; boundary=frame')

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
        ROBOT.SOUND.say("Goodbye")
        ROBOT = None
    return jsonify({'message':'Shutting Down'})

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