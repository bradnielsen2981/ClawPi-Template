/* This is your dashboard JavaScript, it has been embedded into dashboard.html */

//---LOAD AND SHUTDOWN THE ROBOT------------------------------------------------------
function hide_dashboard()
{
    videofeed.innerHTML = ' ';
    load_robot_button.style.display = "block";
    robot_buttons.style.display = "none";
    shutdown_robot_button.style.display = "none";
    camera_fps.textContent = "offline";
    stop_polling();
}
function show_dashboard()
{
    loader.style.display = 'none';
    videofeed.innerHTML = '<img src="/videofeed" class="videofeed" alt="Robot camera">';
    robot_buttons.style.display = "flex";
    shutdown_robot_button.style.display = "block";
    load_robot_button.style.display = "none";
    start_polling();
}

load_robot_button.onclick = load_robot;
function load_robot()
{
    if (robot_loaded == 0)
    {
        load_robot_button.style.display = 'none';
        loader.style.display = 'block';
        new_ajax_helper('/load_robot', load_robot_handler);
    }
}
function load_robot_handler(response) { robot_loaded = 1; show_dashboard(); }

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
    load_robot_button.style.display = "block"; robot_loaded = 0;
}

//---SENDING TO THE SERVER------------------------------------------------------------
// Posts form fields to a Flask route and returns its JSON reply (or {message: ...} if it failed)
async function post(url, fields)
{
    var form = new FormData();
    for (var key in fields) { form.append(key, fields[key]); }
    try {
        var response = await fetch(url, { method: 'POST', body: form });
        return await response.json();
    } catch (error) {
        return { message: 'request failed: ' + error };
    }
}

//---CAMERA TILT AND PAN--------------------------------------------------------------
document.querySelectorAll('.camera-move').forEach(function (button) {
    button.addEventListener('click', async function () {
        var reply = await post('/camera_move', { direction: button.dataset.direction });
        if (reply.pan !== undefined) {
            camera_angles.textContent = 'pan ' + reply.pan + '°  tilt ' + reply.tilt + '°';
        }
    });
});

//---DETECTION BUTTONS----------------------------------------------------------------
// Each detection is a .detect-group: a toggle button plus a dropdown of options (checkboxes or radios).
var detect_groups = {};    // task -> group element
var TASK_NAMES = { colour: 'detect_colour', line: 'detect_line', object: 'detect_object', pose: 'detect_pose' };

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
    var task = group.dataset.task;
    var options = chosen_options(group);
    set_busy(group, true);
    var reply = await post('/set_detection', { task: task, on: on ? '1' : '0', options: options.join(',') });
    set_busy(group, false);
    if (reply.tasks) {
        set_button_state(group, reply.tasks.includes(TASK_NAMES[task]));
    } else {
        console.log(reply.message);
    }
}

function set_all_detection_buttons_off()
{
    Object.values(detect_groups).forEach(function (group) { set_button_state(group, false); });
}

document.querySelectorAll('.detect-group').forEach(function (group) {
    detect_groups[group.dataset.task] = group;
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
            } else if (group.dataset.task === 'pose') {
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
var poll_timer = null;
function start_polling() { if (!poll_timer) { poll_detection_data(); poll_timer = setInterval(poll_detection_data, 500); } }
function stop_polling() { clearInterval(poll_timer); poll_timer = null; }

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
                set_button_state(group, reply.tasks.includes(TASK_NAMES[group.dataset.task]));
            }
        });
        show_detection_data(reply.tasks, reply.data);
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
    detect_pose: function (data) {
        var people = data.people || [];
        if (!people.length) { return nothing(); }
        var parts = [chip(people.length + (people.length == 1 ? ' person' : ' people'), 'chip-found')];
        (data.gestures || []).forEach(function (g) { parts.push(chip(g.replace(/_/g, ' '), 'chip-gesture')); });
        return parts.join(' ');
    },
};
var TASK_LABELS = { detect_colour: ['fa-palette', 'Colour'], detect_line: ['fa-route', 'Line'],
                    detect_object: ['fa-cube', 'Object'], detect_pose: ['fa-person-walking', 'Pose'],
                    detect_depth: ['fa-layer-group', 'Depth'] };

function show_detection_data(tasks, data)
{
    if (!tasks.length) {
        detection_data.innerHTML = '<p class="text-body-secondary mb-0">No detection running.</p>';
    } else {
        detection_data.innerHTML = tasks.map(function (task) {
            var label = TASK_LABELS[task] || ['fa-eye', task];
            var describe = DESCRIBE[task] || function (d) { return escape_html(JSON.stringify(d)); };
            var body = task in data ? describe(data[task]) : '<span class="text-body-secondary">starting...</span>';
            return '<div class="data-row"><span class="data-label"><i class="fa-solid ' + label[0] + ' fa-fw"></i> ' +
                   label[1] + '</span><span class="data-values">' + body + '</span></div>';
        }).join('');
    }
    detection_raw.textContent = JSON.stringify(data, null, 2);
}

//---START----------------------------------------------------------------------------
//hide or show dashboard based on initial value from server on page load
if (robot_loaded == 1) {
    show_dashboard();
} else {
    hide_dashboard();
}
