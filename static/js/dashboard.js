/* This is your dashboard JavaScript, it has been embedded into dashboard.html */

//---LOAD AND SHUTDOWN THE ROBOT------------------------------------------------------
// The camera, cards and controls (#dashboard) only show once the robot has finished loading
function hide_dashboard()
{
    dashboard.style.display = "none";
    videofeed.innerHTML = ' ';
    mapfeed.innerHTML = ' ';
    show_map_status('off');
    end_hold();
    load_robot_button.style.display = "inline-flex";
    robot_buttons.style.display = "none";
    shutdown_robot_button.style.display = "none";
    camera_fps.textContent = "offline";
    stop_polling();
}
function show_dashboard()
{
    loader.style.display = 'none';
    dashboard.style.display = "";
    // A new address each time: with the same address the browser reuses the picture it already has
    // (the last frame before shutdown) instead of opening a new video stream
    videofeed.innerHTML = '<img src="/videofeed?t=' + Date.now() + '" class="videofeed" alt="Robot camera">';
    mapfeed.innerHTML = '<img src="/mapfeed?t=' + Date.now() + '" class="videofeed" alt="Live SLAM map">';
    robot_buttons.style.display = "flex";
    shutdown_robot_button.style.display = "inline-flex";
    load_robot_button.style.display = "none";
    start_polling();
}

load_robot_button.onclick = load_robot;
function load_robot()
{
    if (robot_loaded == 0)
    {
        shutdown_alert.style.setProperty('display', 'none', 'important');
        load_robot_button.style.display = 'none';
        loader.style.display = 'block';
        new_ajax_helper('/load_robot', load_robot_handler);
    }
}
function load_robot_handler(response)
{
    if (response.message == 'robot loaded') {
        robot_loaded = 1;
        show_dashboard();
    } else { // load failed or no longer logged in
        loader.style.display = 'none';
        load_robot_button.style.display = "inline-flex";
        check_logged_in(response);
    }
}

shutdown_robot_button.onclick = shutdown_robot;
function shutdown_robot()
{
    if (robot_loaded == 1)
    {
        hide_dashboard();
        set_all_detection_buttons_off();
        new_ajax_helper('/shutdown_robot', shutdown_robot_handler);
    }
}
function shutdown_robot_handler(response)
{
    load_robot_button.style.display = "inline-flex"; robot_loaded = 0;
}

//---ONE USER AT A TIME--------------------------------------------------------------
// The server logs this browser out if it goes quiet, so keep telling it we're still here
function check_logged_in(response)
{
    if (response && response.message == 'not logged in') { window.location = '/'; }
}
async function heartbeat()
{
    try {
        var response = await fetch('/heartbeat', { method: 'POST' });
        if (response.status == 401) { window.location = '/'; }
    } catch (error) { }
}
setInterval(heartbeat, 15000);

//---SENDING TO THE SERVER------------------------------------------------------------
// Posts form fields to a Flask route and returns its JSON reply (or {message: ...} if it failed)
async function post(url, fields)
{
    var form = new FormData();
    for (var key in fields) { form.append(key, fields[key]); }
    try {
        var response = await fetch(url, { method: 'POST', body: form });
        var reply = await response.json();
        check_logged_in(reply);
        return reply;
    } catch (error) {
        return { message: 'request failed: ' + error };
    }
}

//---CAMERA TILT AND PAN--------------------------------------------------------------
// Tilt and pan are hold-down buttons (below). Centre is a normal click.
function show_camera_angles(reply)
{
    if (reply && reply.pan !== undefined) {
        camera_angles.textContent = 'pan ' + reply.pan + '°  tilt ' + reply.tilt + '°';
    }
}
camera_centre_button.onclick = async function () {
    end_hold();
    show_camera_angles(await post('/camera_centre', {}));
};

//---HOLD-DOWN BUTTONS (movement, claw, camera)---------------------------------------
// While a .hold-btn is held its action is sent every HOLD_RESEND_MS. Letting go sends /release (stop).
// If the robot hears nothing for 3 seconds it stops by itself, so a lost release can't leave it running.
var HOLD_RESEND_MS = 500;
var held = null;              // {button, timer} of the button being held
var hold_seq = Date.now();    // numbers every hold/release so the robot ignores one that arrives late

async function send_hold(url, fields)
{
    fields.seq = ++hold_seq;
    var reply = await post(url, fields);
    show_camera_angles(reply);
    return reply;
}

function start_hold(button)
{
    if (robot_loaded != 1 || (held && held.button === button)) { return; }
    if (held) { clearInterval(held.timer); held.button.classList.remove('holding'); }
    var action = button.dataset.action;
    button.classList.add('holding');
    held = { button: button, timer: setInterval(function () { send_hold('/hold', { action: action }); }, HOLD_RESEND_MS) };
    send_hold('/hold', { action: action });
}

function end_hold()
{
    if (!held) { return; }
    clearInterval(held.timer);
    held.button.classList.remove('holding');
    held = null;
    send_hold('/release', {});
}

document.querySelectorAll('.hold-btn').forEach(function (button) {
    // Mouse, touch and pen. Capturing the pointer means the release is seen even off the button.
    button.addEventListener('pointerdown', function (event) {
        if (event.button !== 0) { return; }
        event.preventDefault();
        button.setPointerCapture(event.pointerId);
        start_hold(button);
    });
    ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(function (name) {
        button.addEventListener(name, function () { if (held && held.button === button) { end_hold(); } });
    });
    button.addEventListener('contextmenu', function (event) { event.preventDefault(); }); // long press on a phone
    // Keyboard: hold Space or Enter on a focused button
    button.addEventListener('keydown', function (event) {
        if (event.key === ' ' || event.key === 'Enter') { event.preventDefault(); if (!event.repeat) { start_hold(button); } }
    });
    button.addEventListener('keyup', function (event) {
        if ((event.key === ' ' || event.key === 'Enter') && held && held.button === button) { end_hold(); }
    });
});

// Arrow keys pan and tilt the camera for as long as they are held
var ARROW_KEYS = { ArrowUp: 'camera_up', ArrowDown: 'camera_down', ArrowLeft: 'camera_left', ArrowRight: 'camera_right' };
function typing_in(target) { return target.closest('input, select, textarea, [contenteditable]'); }
document.addEventListener('keydown', function (event) {
    var action = ARROW_KEYS[event.key];
    if (!action || robot_loaded != 1 || typing_in(event.target)) { return; }
    event.preventDefault(); // don't scroll the page
    if (!event.repeat) { start_hold(document.querySelector('.hold-btn[data-action="' + action + '"]')); }
});
document.addEventListener('keyup', function (event) {
    var action = ARROW_KEYS[event.key];
    if (action && held && held.button.dataset.action === action) { end_hold(); }
});

// Stop if the window loses focus or the tab is hidden while something is held
window.addEventListener('blur', end_hold);
document.addEventListener('visibilitychange', function () { if (document.hidden) { end_hold(); } });

move_stop_button.onclick = function () { end_hold(); post('/stop', {}); };

//---DETECTION BUTTONS----------------------------------------------------------------
// Each detection is a .detect-group: a toggle button plus a dropdown of options (checkboxes or radios).
var detect_groups = {};    // detector -> group element
var DETECTOR_NAMES = { colour: 'detect_colour', line: 'detect_line', object: 'detect_object', pose: 'detect_pose',
                       face: 'detect_face' };

function chosen_options(group)
{
    return Array.from(group.querySelectorAll('.dropdown-menu input:checked'))
                .filter(function (input) { return input.value && !input.classList.contains('pose-all'); })
                .map(function (input) { return input.value; });
}

function set_button_state(group, on)
{
    group.dataset.on = on ? '1' : '0';
    group.querySelectorAll('.btn').forEach(function (button) {
        button.classList.toggle('btn-success', on);
        button.classList.toggle('btn-outline-secondary', !on);
    });
}

function set_busy(group, busy)
{
    group.dataset.busy = busy ? '1' : '0';
    var toggle = group.querySelector('.detect-toggle');
    toggle.disabled = busy;
    var spinner = toggle.querySelector('.spinner-border');
    if (busy && !spinner) {
        toggle.insertAdjacentHTML('beforeend', '<span class="spinner-border spinner-border-sm ms-2" aria-hidden="true"></span>');
    } else if (!busy && spinner) {
        spinner.remove();
    }
}

// Sends the group's state to the server. Turning on with nothing chosen (colours, line) turns it off.
async function send_detection(group, on)
{
    if (robot_loaded != 1) { return; }
    var detector = group.dataset.detector;
    var options = chosen_options(group);
    set_busy(group, true);
    var reply = await post('/set_detection', { detector: detector, on: on ? '1' : '0', options: options.join(',') });
    set_busy(group, false);
    if (reply.detectors) {
        set_button_state(group, reply.detectors.includes(DETECTOR_NAMES[detector]));
    } else {
        console.log(reply.message);
    }
}

function set_all_detection_buttons_off()
{
    Object.values(detect_groups).forEach(function (group) { set_button_state(group, false); });
}

document.querySelectorAll('.detect-group').forEach(function (group) {
    detect_groups[group.dataset.detector] = group;
    set_button_state(group, false);

    // The main button turns the detection on or off
    group.querySelector('.detect-toggle').addEventListener('click', function () {
        send_detection(group, group.dataset.on !== '1');
    });

    // Changing the options while it is on updates the robot straight away
    group.querySelectorAll('.dropdown-menu input').forEach(function (input) {
        input.addEventListener('change', function () {
            if (input.classList.contains('pose-all')) {
                group.querySelectorAll('.dropdown-menu input[value]').forEach(function (box) { box.checked = input.checked; });
            } else if (group.dataset.detector === 'pose') {
                var boxes = Array.from(group.querySelectorAll('.dropdown-menu input[value]'));
                group.querySelector('.pose-all').checked = boxes.every(function (box) { return box.checked; });
            }
            if (group.dataset.on === '1') { send_detection(group, true); }
        });
    });
});

stop_detection_button.onclick = function () {
    new_ajax_helper('/turn_off_detection', function () { set_all_detection_buttons_off(); });
};

//---DETECTION DATA CARD--------------------------------------------------------------
var poll_timer = null, sensor_timer = null, map_timer = null;
function start_polling()
{
    if (!poll_timer) { poll_detection_data(); poll_timer = setInterval(poll_detection_data, 500); }
    if (!sensor_timer) { poll_sensor_data(); sensor_timer = setInterval(poll_sensor_data, 1000); }
    if (!map_timer) { poll_map_status(); map_timer = setInterval(poll_map_status, 2000); }
}
function stop_polling()
{
    clearInterval(poll_timer); poll_timer = null;
    clearInterval(sensor_timer); sensor_timer = null;
    clearInterval(map_timer); map_timer = null;
}

async function poll_detection_data()
{
    try {
        var response = await fetch('/detection_data');
        var reply = await response.json();
        if (!reply.loaded) { return; }
        camera_fps.textContent = reply.fps + ' fps';
        // Keep the buttons in step with what the robot is really doing (unless a change is being sent)
        Object.values(detect_groups).forEach(function (group) {
            if (group.dataset.busy !== '1') {
                set_button_state(group, reply.detectors.includes(DETECTOR_NAMES[group.dataset.detector]));
            }
        });
        show_detection_data(reply.detectors, reply.data);
    } catch (error) {
        camera_fps.textContent = 'offline';
    }
}

function escape_html(text)
{
    var div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}
function chip(text, kind) { return '<span class="data-chip ' + (kind || '') + '">' + escape_html(text) + '</span>'; }
function nothing() { return '<span class="text-body-secondary">nothing found</span>'; }

// Turns each detector's data into a short readable line
var DESCRIBE = {
    detect_colour: function (data) {
        var colours = Object.keys(data);
        if (!colours.length) { return nothing(); }
        return colours.map(function (colour) {
            var c = data[colour], centre = c.rect ? c.rect[0] : ['?', '?'];
            var text = colour + ' at (' + centre[0] + ', ' + centre[1] + ') area ' + c.area;
            if (c.distance != null) { text += ' · ' + c.distance + 'mm'; }
            return chip(text, 'chip-found');
        }).join(' ');
    },
    detect_line: function (data) {
        if (!data.found) { return nothing(); }
        var parts = [];
        if (data.angle !== undefined) { parts.push(chip('angle ' + data.angle + '°', 'chip-found'), chip('offset ' + data.offset + 'px')); }
        if (data.segments && data.segments.length) { parts.push(chip(data.segments.length + ' straight lines', 'chip-found')); }
        if (data.crossing) { parts.push(chip('intersection at y=' + data.crossing.y, 'chip-found')); }
        return parts.join(' ');
    },
    detect_object: function (data) {
        var targets = data.targets || [];
        if (!targets.length) { return nothing(); }
        return targets.map(function (t) {
            var text = t.class + ' ' + Math.round(t.score * 100) + '%';
            if (t.distance != null) { text += ' · ' + t.distance + 'mm'; }
            return chip(text, 'chip-found');
        }).join(' ');
    },
    detect_face: function (data) {
        var faces = data.faces || [];
        if (!faces.length) { return nothing(); }
        return faces.map(function (f) {
            return chip('face at (' + f.center[0] + ', ' + f.center[1] + ') ' + Math.round(f.score * 100) + '%', 'chip-found');
        }).join(' ');
    },
    detect_pose: function (data) {
        var people = data.people || [];
        if (!people.length) { return nothing(); }
        var parts = [chip(people.length + (people.length == 1 ? ' person' : ' people'), 'chip-found')];
        (data.gestures || []).forEach(function (g) { parts.push(chip(g.replace(/_/g, ' '), 'chip-gesture')); });
        return parts.join(' ');
    },
};
var DETECTOR_LABELS = { detect_colour: ['fa-palette', 'Colour'], detect_line: ['fa-route', 'Line'],
                    detect_object: ['fa-cube', 'Object'], detect_pose: ['fa-person-walking', 'Pose'],
                    detect_face: ['fa-face-smile', 'Face'], detect_depth: ['fa-layer-group', 'Depth'] };

function show_detection_data(detectors, data)
{
    if (!detectors.length) {
        detection_data.innerHTML = '<p class="text-body-secondary mb-0">No detection running.</p>';
    } else {
        detection_data.innerHTML = detectors.map(function (detector) {
            var label = DETECTOR_LABELS[detector] || ['fa-eye', detector];
            var describe = DESCRIBE[detector] || function (d) { return escape_html(JSON.stringify(d)); };
            var body = detector in data ? describe(data[detector]) : '<span class="text-body-secondary">starting...</span>';
            return '<div class="data-row"><span class="data-label"><i class="fa-solid ' + label[0] + ' fa-fw"></i> ' +
                   label[1] + '</span><span class="data-values">' + body + '</span></div>';
        }).join('');
    }
    detection_raw.textContent = JSON.stringify(data, null, 2);
}

//---SENSORS CARD---------------------------------------------------------------------
var LOW_VOLTAGE = 7.0; // volts, the chip turns red below this

async function poll_sensor_data()
{
    try {
        var response = await fetch('/sensor_data');
        var reply = await response.json();
        if (reply.loaded) { show_sensor_data(reply); }
        else if (robot_loaded == 1 && reply.shutdown_reason) { robot_shut_itself_down(reply.shutdown_reason); }
    } catch (error) { }
}

// The robot shut itself down (e.g. a routine crashed): show why, and offer Load Robot again
function robot_shut_itself_down(reason)
{
    hide_dashboard();
    set_all_detection_buttons_off();
    robot_loaded = 0;
    shutdown_reason.textContent = reason;
    shutdown_alert.style.setProperty('display', 'flex', 'important');
}

function show_sensor_data(reply)
{
    var v = reply.voltage;
    sensor_voltage.textContent = v == null ? 'no reading' : v.toFixed(2) + ' V';
    sensor_voltage.className = 'data-chip' + (v != null && v < LOW_VOLTAGE ? ' chip-low' : '');

    var imu = reply.imu;
    if (!imu) {
        sensor_imu.innerHTML = '<span class="text-body-secondary">no reading</span>';
    } else {
        var deg = function (x) { return Math.round(x) + '°'; };
        var xyz = function (a) { return a.map(function (x) { return x.toFixed(1); }).join(', '); };
        sensor_imu.innerHTML = chip('roll ' + deg(imu.rpy[0])) + ' ' + chip('pitch ' + deg(imu.rpy[1])) + ' ' +
                               chip('yaw ' + deg(imu.rpy[2])) + ' ' + chip('accel ' + xyz(imu.accel) + ' m/s²') + ' ' +
                               chip('gyro ' + xyz(imu.gyro) + ' rad/s');
    }

    show_line_follower(reply.line);
}

// The 4 channel line follower: line.sensors is [far left, centre left, centre right, far right], true = sees the line
var SENSOR_NAMES = ['far left', 'centre left', 'centre right', 'far right'];
function show_line_follower(line)
{
    var sensors = line.sensors;
    line_sensors.querySelectorAll('.line-sensor').forEach(function (light, i) {
        light.classList.toggle('on', !!(sensors && sensors[i]));
    });
    if (!sensors) {
        line_sensors.setAttribute('aria-label', 'Line follower: no reading');
        sensor_line.innerHTML = '<span class="text-body-secondary">' + escape_html(line.error || 'no reading') + '</span>';
        return;
    }
    var seen = SENSOR_NAMES.filter(function (name, i) { return sensors[i]; });
    line_sensors.setAttribute('aria-label', 'Line follower: ' + (seen.length ? 'line under ' + seen.join(', ') : 'no line'));
    if (!line.on_line) {
        sensor_line.innerHTML = chip('no line under the robot');
        return;
    }
    var p = line.position, where = Math.abs(p) < 0.2 ? 'centred' : (p < 0 ? 'line to the left' : 'line to the right');
    var parts = [chip(where + ' (' + p.toFixed(2) + ')', 'chip-found')];
    if (line.junction) { parts.push(chip('junction', 'chip-gesture')); }
    sensor_line.innerHTML = parts.join(' ');
}

//---SLAM MAP CARD-------------------------------------------------------------------
// The map picture is a live stream (/mapfeed). The button starts and stops the SLAM program.
var map_state = 'off';   // off, starting, mapping, stopping or failed
function show_map_status(state)
{
    map_state = state;
    var looks = {
        off:      ['off',        'text-bg-secondary', 'fa-play', 'Start', 'btn-outline-success', 'Start mapping'],
        starting: ['starting...', 'text-bg-warning',  'fa-spinner fa-spin', 'Start', 'btn-outline-success', 'Starting mapping'],
        mapping:  ['mapping',    'text-bg-success',   'fa-stop', 'Stop', 'btn-outline-danger', 'Stop mapping'],
        stopping: ['stopping...', 'text-bg-warning',  'fa-spinner fa-spin', 'Stop', 'btn-outline-danger', 'Stopping mapping'],
        failed:   ['failed',     'text-bg-danger',    'fa-play', 'Start', 'btn-outline-success', 'Start mapping']
    }[state];
    map_status.textContent = looks[0];
    map_status.className = 'badge rounded-pill ms-auto ' + looks[1];
    map_toggle_button.innerHTML = '<i class="fa-solid ' + looks[2] + '" aria-hidden="true"></i> ' + looks[3];
    map_toggle_button.className = 'btn btn-sm ' + looks[4];
    map_toggle_button.setAttribute('aria-label', looks[5]);
    map_toggle_button.title = looks[5];
    map_toggle_button.disabled = (state == 'starting' || state == 'stopping');
}

map_toggle_button.onclick = async function () {
    if (robot_loaded != 1 || map_toggle_button.disabled) { return; }
    if (map_state == 'mapping') {
        show_map_status('stopping');
        await post('/stop_mapping', {});
        show_map_status('off');
    } else {
        show_map_status('starting'); // can take up to 30 s while the SLAM program starts
        var reply = await post('/start_mapping', {});
        show_map_status(reply.status == 'Mapping' ? 'mapping' : 'failed');
    }
};

async function poll_map_status()
{
    if (map_state == 'starting' || map_state == 'stopping') { return; } // a change is being sent
    try {
        var response = await fetch('/map_status');
        var reply = await response.json();
        if (!reply.loaded) { return; }
        if (reply.status == 'Mapping') { show_map_status('mapping'); }
        else if (map_state == 'mapping') { show_map_status('off'); }
    } catch (error) { }
}

//---START----------------------------------------------------------------------------
//hide or show dashboard based on initial value from server on page load
if (robot_loaded == 1) {
    show_dashboard();
} else {
    hide_dashboard();
}
