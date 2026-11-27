# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# Voice agent for the ClawPi's WonderEcho Pro AI voice box.
#
# The WonderEcho Pro plugs into USB and shows up as two things:
#   - a USB sound card (ALSA card 'Device'): its speaker and noise-cancelling microphone
#   - a CH340 serial port (/dev/ring_mic, 115200 baud): its own chip listens for the wake word
#     "Hello Hiwonder" and sends the 5 bytes AA 55 03 00 FB when it hears it
#     (other packets AA 55 xx yy FB are the chip's built-in command words)
# See https://wiki.hiwonder.com/projects/ClawPi/en/latest/docs/12.%20Large%20AI%20Model%20Course.html
#
# This does the same job as Hiwonder's voice nodes (vocal_detect -> agent_process -> llm_control -> tts_node)
# in one Python object, with no ROS needed for the voice side:
#
#   wake word (chip) -> record what is said -> speech to text -> understand the command -> do it -> speak the reply
#
# Each step has backups, so the agent still works without an internet API key:
#   Speech to text: OpenAI Whisper (API key)  -> Vosk (offline, if installed)          -> none (wake word + chip commands only)
#   Understanding:  OpenAI chat + tools (key) -> Ollama local model (if running)       -> built-in keyword commands
#   Text to speech: OpenAI voice (API key)    -> Piper (offline) -> espeak-ng (offline) -> printed text
# Sounds and music play with aplay/ffmpeg (no pygame needed).
#
# Quick start:
#   VOICE = VoiceAgentInterface(robot=ROBOT)   # robot is optional (a ClawPi or RobotInterface)
#   VOICE.set_api_key("sk-...")                # optional, see set_api_key()
#   VOICE.say("Hello")
#   VOICE.start_listening()                    # say "Hello Hiwonder", then a command like "move forward two seconds"
#   ...
#   VOICE.stop()

import os, sys, re, io, json, time, math, wave, queue, base64, signal, shutil, logging, threading, subprocess, tempfile
from collections import deque
from datetime import datetime

import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from loggerinterface import get_logger

# --- WonderEcho Pro hardware ---------------------------------------------------------------
ALSA_DEVICE = 'plughw:CARD=Device,DEV=0'   # The WonderEcho's sound card (speaker and microphone)
MIXER_CARD = 'Device'                      # For volume: amixer -c Device sset Speaker 80%
MIXER_CONTROL = 'Speaker'
SERIAL_PORT = '/dev/ring_mic'              # The WonderEcho's CH340 serial port (wake word / command words)
SERIAL_BAUD = 115200
WAKE_PACKET = (0x03, 0x00)                 # AA 55 03 00 FB: "Hello Hiwonder" heard
SLEEP_PACKET = (0x02, 0x00)                # AA 55 02 00 FB: the chip went back to sleep
MIC_RATE = 16000                           # Microphone recording: 16 kHz, mono, 16 bit
CHUNK = 480                                # Samples per microphone chunk (30 ms)

# The chip's built-in command words (from Hiwonder's "Command Words and Response Protocol List").
# Fill in names here as you find the codes (each one is printed when heard): {(0x01, 0x00): 'forward', ...}
CHIP_COMMANDS = {}

# Hiwonder's English prompt sounds ("I'm here", a "dong", ...)
HIWONDER_AUDIO = '/home/ubuntu/ros2_ws/src/large_models/large_models/large_models/resources/audio/en'
HIWONDER_CONFIG = '/home/ubuntu/ros2_ws/src/large_models/large_models/large_models/config.py'
WAKE_SOUND = os.path.join(HIWONDER_AUDIO, 'wakeup.wav')      # "I'm here"
HEARD_SOUND = os.path.join(HIWONDER_AUDIO, 'dong.wav')       # Finished listening
NO_VOICE_SOUND = os.path.join(HIWONDER_AUDIO, 'no_voice.wav') # Didn't hear anything

# --- API key and models ---------------------------------------------------------------------
# Saved outside the project so it never ends up in git
API_KEY_FILE = '/home/ubuntu/.config/clawpi/openai_api_key'
OPENAI_BASE_URL = 'https://api.openai.com/v1'
LLM_MODEL = 'gpt-4o-mini'      # Chat model that understands commands (same default as Hiwonder's config.py)
VISION_MODEL = 'gpt-4o'        # Answers questions about the camera picture
TTS_MODEL = 'tts-1'
TTS_VOICE = 'onyx'
STT_MODEL = 'whisper-1'

# --- Offline backups (used when they are installed) -------------------------------------------
OLLAMA_URL = 'http://localhost:11434/v1'   # Ollama's OpenAI-style endpoint
OLLAMA_MODEL = 'qwen3:1.7b'                # Hiwonder's suggested offline model
PIPER_VOICE = '/home/ubuntu/.local/share/piper/en_GB-alan-medium.onnx'  # Piper voice file (.onnx + .onnx.json)
VOSK_MODEL = '/home/ubuntu/.local/share/vosk/vosk-model-small-en-us-0.15'
ESPEAK_VOICE = 'en-gb'

ROBOT_NAME = 'Wally'
SYSTEM_PROMPT = """You are {name}, a small friendly robot (a Hiwonder ClawPi) with mecanum wheels, a camera on a
pan/tilt head and a claw on a lifting arm. People talk to you through your microphone and hear your replies from
a speaker, so answer in one or two short, plain sentences with no lists, markdown or emoji.
When the user asks you to do something, call the matching tool, then say briefly what you did.
Only call tools for things the user actually asked for. If a request is unsafe or you have no tool for it, say so.
Positive turn angles are to the left (anticlockwise), negative to the right."""

NUMBER_WORDS = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
                'eight': 8, 'nine': 9, 'ten': 10, 'half': 0.5, 'twenty': 20, 'thirty': 30, 'forty': 40,
                'forty five': 45, 'fifty': 50, 'sixty': 60, 'ninety': 90, 'a hundred': 100, 'one hundred': 100,
                'one hundred and eighty': 180, 'a hundred and eighty': 180, 'two hundred and seventy': 270,
                'three hundred and sixty': 360}


def has_program(name):
    """True if a command line program (e.g. 'espeak-ng') is installed."""
    return shutil.which(name) is not None

def wav_bytes(samples, rate=MIC_RATE):
    """Wraps raw 16 bit mono samples (bytes) in a WAV file (bytes)."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples)
    return buffer.getvalue()

def chunk_level(chunk):
    """Loudness (RMS) of a chunk of 16 bit samples. Quiet room ~100-400, talking ~1000-5000, clap 8000+."""
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0

def parse_number(text, default=None):
    """First number in text, digits or words: '2.5 seconds' -> 2.5, 'two seconds' -> 2, 'half a metre' -> 0.5."""
    match = re.search(r'-?\d+(?:\.\d+)?', text)
    if match:
        return float(match.group())
    for words in sorted(NUMBER_WORDS, key=len, reverse=True):
        if re.search(rf'\b{words}\b', text):
            return float(NUMBER_WORDS[words])
    return default


#-------------------------------------------
# The microphone: one arecord process streaming 30 ms chunks
class MicStream():
    def __init__(self, device=ALSA_DEVICE, rate=MIC_RATE):
        self.process = subprocess.Popen(
            ['arecord', '-D', device, '-f', 'S16_LE', '-r', str(rate), '-c', '1', '-t', 'raw', '-q'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.chunk_bytes = CHUNK * 2

    def read(self):
        """The next 30 ms chunk (bytes), waiting for it. Empty if the microphone stopped."""
        return self.process.stdout.read(self.chunk_bytes) or b''

    def flush(self):
        """Throws away audio that built up (e.g. while the robot was talking), so it isn't heard as speech."""
        os.set_blocking(self.process.stdout.fileno(), False)
        try:
            while self.process.stdout.read(65536):
                pass
        except (BlockingIOError, TypeError):
            pass
        finally:
            os.set_blocking(self.process.stdout.fileno(), True)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


#-------------------------------------------
# The Voice Agent Object
class VoiceAgentInterface():

    def __init__(self, robot=None, name=ROBOT_NAME, api_key=None, volume=None):
        """
        Sets up the voice agent. Nothing listens until start_listening() is called.

        Args:
            robot (ClawPi or RobotInterface, optional): The robot to control with voice commands. Without one the
                                                        agent still talks and understands, but can't move.
            name (str, optional): The robot's name, used in its personality.
            api_key (str, optional): An OpenAI API key. If None, one is looked for (see load_api_key()).
            volume (int, optional): Speaker volume 0-100. None leaves it as it is.
        """
        self.robot = robot
        self.name = name
        self.logger = get_logger('Voice') # logs/voice.log

        # API / model settings (see set_api_key, use_local_llm)
        self.api_key = None
        self.base_url = OPENAI_BASE_URL
        self.llm_model = LLM_MODEL
        self._client = None
        self.load_api_key(api_key)

        # Speaking: say() queues text, one background thread speaks it in order
        self._speech_queue = queue.Queue()
        self._speaking = threading.Event()
        self._sound_processes = []  # The aplay (and ffmpeg) processes playing right now
        self._sound_lock = threading.Lock()
        threading.Thread(target=self._speech_loop, daemon=True, name='voice-speech').start()

        # Listening
        self.listening = False
        self.wake_word = True             # True = wait for "Hello Hiwonder"; False = respond to any speech
        self.respond = True               # True = do the command and speak the reply; False = only report the text
        self.sound_level = 0.0            # Microphone loudness, updated while listening
        self.loud_threshold = 8000        # Loudness that counts as a loud sound (clap, bang)
        self.silence_seconds = 1.0        # Quiet time that ends a sentence
        self.max_record_seconds = 8.0
        self.start_timeout = 5.0          # Seconds to start talking after the wake word
        self.last_heard = ''              # The newest thing understood
        self.last_reply = ''
        self.last_chip_command = None     # The newest built-in command word code, e.g. (0x01, 0x00)
        self.history = deque(maxlen=20)   # [{'time', 'heard', 'reply', 'actions'}] newest last
        self.on_wake = None               # Callbacks, see start_listening()
        self.on_heard = None
        self.on_loud_sound = None
        self.on_chip_command = None
        self._listen_thread = None
        self._serial = None
        self._brain_lock = threading.Lock()
        self._conversation = deque(maxlen=12)  # Recent chat turns, so follow-up questions make sense

        # Commands the agent can carry out: name -> {'function', 'description', 'parameters'}
        self.commands = {}
        self._add_default_commands()

        self._vosk_model = None
        if volume is not None:
            self.set_volume(volume)
        self.logger.info(f"Voice agent ready: {self.describe_backends()}")

    #-------------------------------------------
    # API keys and models
    def load_api_key(self, api_key=None):
        """
        Finds an OpenAI API key, in this order: the api_key given, the OPENAI_API_KEY environment variable, the
        file saved by set_api_key(save=True), then Hiwonder's config.py (llm_api_key). Without one, the offline
        backups are used.

        Returns:
            bool: True if a key was found.
        """
        key = api_key or os.environ.get('OPENAI_API_KEY', '')
        if not key and os.path.exists(API_KEY_FILE):
            with open(API_KEY_FILE) as f:
                key = f.read().strip()
        if not key and os.path.exists(HIWONDER_CONFIG):
            with open(HIWONDER_CONFIG) as f:
                match = re.search(r"^llm_api_key\s*=\s*['\"]([^'\"]+)['\"]", f.read(), re.MULTILINE)
            key = match.group(1) if match else ''
        self.set_api_key(key or None)
        return bool(key)

    def set_api_key(self, api_key, base_url=OPENAI_BASE_URL, llm_model=LLM_MODEL, save=False):
        """
        Sets the OpenAI API key used for speech to text, understanding and the voice.
        Get a key at https://platform.openai.com/api-keys (usage is billed by OpenAI).

        Args:
            api_key (str or None): The key ("sk-..."). None or '' switches back to the offline backups.
            base_url (str, optional): Another OpenAI-compatible service, e.g. 'https://openrouter.ai/api/v1'.
            llm_model (str, optional): Chat model name.
            save (bool, optional): True saves the key (outside the project, readable only by its owner) so it is
                                   found next time without being in your code.
        """
        self.api_key = api_key or None
        self.base_url = base_url
        self.llm_model = llm_model
        self._client = None
        if save and api_key:
            os.makedirs(os.path.dirname(API_KEY_FILE), exist_ok=True)
            with open(API_KEY_FILE, 'w') as f:
                f.write(api_key.strip())
            os.chmod(API_KEY_FILE, 0o600)

    def use_local_llm(self, model=OLLAMA_MODEL, url=OLLAMA_URL):
        """
        Understands commands with a model running on the robot through Ollama (https://ollama.com) instead of
        OpenAI. Speech to text and the voice still use OpenAI if a key was set, otherwise their offline backups.

        Returns:
            bool: True if Ollama answered.
        """
        if not self._ollama_running(url):
            self.logger.warning(f"Ollama isn't running at {url}")
            return False
        self._local_llm = (url, model)
        return True

    def _ollama_running(self, url=OLLAMA_URL):
        try:
            import requests
            return requests.get(url.replace('/v1', '/api/tags'), timeout=0.5).ok
        except Exception:
            return False

    def _openai(self):
        """The OpenAI client for the API key, or None without one."""
        if not self.api_key:
            return None
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30)
        return self._client

    def _chat_client(self):
        """(client, model) for understanding commands: OpenAI, else a local Ollama model, else (None, None)."""
        local = getattr(self, '_local_llm', None)
        if local:
            from openai import OpenAI
            return OpenAI(api_key='ollama', base_url=local[0], timeout=60), local[1]
        if self.api_key:
            return self._openai(), self.llm_model
        if self._ollama_running():
            self._local_llm = (OLLAMA_URL, OLLAMA_MODEL)
            return self._chat_client()
        return None, None

    def describe_backends(self):
        """
        Which engine each step will use right now.

        Returns:
            dict: {'speech_to_text', 'understanding', 'voice'}, e.g. {'speech_to_text': 'vosk', ...}
        """
        stt = 'openai' if self.api_key else ('vosk' if self._vosk_available() else 'none')
        local = getattr(self, '_local_llm', None)
        brain = ('ollama' if local else 'openai') if (local or self.api_key) else (
            'ollama' if self._ollama_running() else 'keywords')
        voice = 'openai' if self.api_key else (
            'piper' if self._piper_available() else 'espeak-ng' if has_program('espeak-ng') else 'print')
        return {'speech_to_text': stt, 'understanding': brain, 'voice': voice}

    #-------------------------------------------
    # Speaker: sounds, music and volume
    def play_sound(self, path, block=False, loops=1):
        """
        Plays a sound or music file (wav, mp3, ogg, ...) on the WonderEcho speaker. Stops anything already playing.

        Args:
            path (str): The file, e.g. 'static/music/missionimpossible.mp3'.
            block (bool, optional): True waits until it has finished.
            loops (int, optional): How many times to play it. -1 repeats until stop_sound().
        """
        if not os.path.exists(path):
            self.logger.warning(f"Sound file not found: {path}")
            return
        self.stop_sound()
        if path.lower().endswith('.wav') and loops == 1:
            processes = [subprocess.Popen(['aplay', '-q', '-D', ALSA_DEVICE, path], stderr=subprocess.DEVNULL)]
        else:  # ffmpeg decodes anything (and loops), aplay plays it
            decoder = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-stream_loop', str(loops - 1 if loops > 0 else -1),
                                        '-i', path, '-f', 'wav', '-ac', '2', '-ar', '44100', '-'],
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            player = subprocess.Popen(['aplay', '-q', '-D', ALSA_DEVICE], stdin=decoder.stdout, stderr=subprocess.DEVNULL)
            decoder.stdout.close()  # aplay owns the pipe now
            processes = [decoder, player]
        with self._sound_lock:
            self._sound_processes = processes
        if block:
            processes[-1].wait()
            for p in processes:
                p.wait()

    def _signal_sound(self, sig):
        """Sends a signal to the processes playing sound. Returns them, or [] if nothing is playing."""
        with self._sound_lock:
            processes = list(getattr(self, '_sound_processes', []))
        if not any(p.poll() is None for p in processes):
            return []
        for p in processes:
            if p.poll() is None:
                try:
                    p.send_signal(sig)
                except ProcessLookupError:
                    pass
        return processes

    def stop_sound(self):
        """Stops the sound or music that is playing."""
        processes = self._signal_sound(signal.SIGCONT)  # A paused process can't finish stopping
        if processes:
            self._signal_sound(signal.SIGTERM)
            for p in processes:
                try:
                    p.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()

    def pause_sound(self):
        """Pauses the sound or music (resume_sound() carries on)."""
        self._signal_sound(signal.SIGSTOP)

    def resume_sound(self):
        """Carries on after pause_sound()."""
        self._signal_sound(signal.SIGCONT)

    def is_playing(self):
        """True while a sound or music file is playing."""
        with self._sound_lock:
            return any(p.poll() is None for p in getattr(self, '_sound_processes', []))

    # The old SoundInterface names, so existing code keeps working
    def load_mp3(self, song):
        self._music = song
    def play_music(self, times=-1):
        self.play_sound(getattr(self, '_music', ''), loops=times)
    pause_music, unpause_music, stop_music = pause_sound, resume_sound, stop_sound

    def set_volume(self, percent=80):
        """
        Sets the WonderEcho speaker volume.

        Args:
            percent (float): 0-100. Values 0.0-1.0 are treated as fractions (the old SoundInterface style).
        """
        if 0 < percent <= 1:
            percent *= 100
        percent = int(max(0, min(100, percent)))
        subprocess.run(['amixer', '-q', '-c', MIXER_CARD, 'sset', MIXER_CONTROL, f'{percent}%'],
                       stderr=subprocess.DEVNULL)

    def get_volume(self):
        """The speaker volume 0-100, or None if it can't be read."""
        result = subprocess.run(['amixer', '-c', MIXER_CARD, 'sget', MIXER_CONTROL], capture_output=True, text=True)
        match = re.search(r'\[(\d+)%\]', result.stdout)
        return int(match.group(1)) if match else None

    #-------------------------------------------
    # Speaking (text to speech)
    def say(self, text, block=False):
        """
        Speaks text out loud. Sentences are spoken in order, in the background.

        Args:
            text (str): What to say.
            block (bool, optional): True waits until it has been said.
        """
        text = str(text or '').strip()
        if not text:
            return
        done = threading.Event()
        self._speaking.set()
        self._speech_queue.put((text, done))
        if block:
            done.wait()

    def is_speaking(self):
        """True while there is speech waiting or being spoken."""
        return self._speaking.is_set()

    def stop_speaking(self):
        """Drops any speech still waiting and stops the sentence being spoken."""
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()[1].set()
            except queue.Empty:
                break
        self.stop_sound()

    def _speech_loop(self):
        while True:
            text, done = self._speech_queue.get()
            try:
                self.logger.info(f"say: {text}")
                path = self._synthesise(text)
                if path:
                    self.play_sound(path, block=True)
                    os.remove(path)
                else:
                    print(f"[{self.name} says] {text}")
            except Exception as e:
                print(f"[{self.name} says] {text}   (voice failed: {e})")
                self.logger.error(f"voice failed: {e}")
            finally:
                done.set()
                if self._speech_queue.empty():
                    self._speaking.clear()

    def _synthesise(self, text):
        """Turns text into a temporary WAV file with the best voice available. Returns its path, or None."""
        path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        client = self._openai()
        if client:
            try:
                audio = client.audio.speech.create(model=TTS_MODEL, voice=TTS_VOICE, input=text, response_format='wav')
                with open(path, 'wb') as f:
                    f.write(audio.content)
                return path
            except Exception as e:
                self.logger.warning(f"OpenAI voice failed, using a backup voice: {e}")
        if self._piper_available():
            result = subprocess.run([sys.executable, '-m', 'piper', '--model', PIPER_VOICE, '--output_file', path],
                                    input=text, text=True, capture_output=True)
            if result.returncode == 0:
                return path
        if has_program('espeak-ng'):
            subprocess.run(['espeak-ng', '-v', ESPEAK_VOICE, '-s', '150', '-w', path, text])
            return path
        os.remove(path)
        return None

    def _piper_available(self):
        if not hasattr(self, '_has_piper'):
            try:
                import importlib.util
                self._has_piper = importlib.util.find_spec('piper') is not None and os.path.exists(PIPER_VOICE)
            except Exception:
                self._has_piper = False
        return self._has_piper

    #-------------------------------------------
    # Microphone: recording and speech to text
    def record(self, seconds=5, path='static/recording.wav'):
        """
        Records the microphone to a WAV file. Not while listening (the microphone is busy then).

        Returns:
            str: The file path.
        """
        subprocess.run(['arecord', '-q', '-D', ALSA_DEVICE, '-f', 'S16_LE', '-r', str(MIC_RATE), '-c', '1',
                        '-d', str(int(math.ceil(seconds))), path])
        return path

    def listen_once(self, timeout=None):
        """
        Waits for someone to say something, then returns the words (no wake word needed). Blocks until done.
        Not while start_listening() is running.

        Returns:
            str: The words heard ('' if nothing was heard or there is no speech to text available).
        """
        mic = MicStream()
        try:
            audio = self._record_sentence(mic, timeout or self.start_timeout)
        finally:
            mic.close()
        return self.speech_to_text(audio) if audio else ''

    def speech_to_text(self, audio):
        """
        Turns recorded speech into words: OpenAI Whisper with an API key, otherwise Vosk if installed.

        Args:
            audio (bytes): 16 kHz mono 16 bit samples, or a whole WAV file.

        Returns:
            str: The words ('' if none, or no speech to text available).
        """
        if not audio.startswith(b'RIFF'):
            audio = wav_bytes(audio)
        client = self._openai()
        if client:
            try:
                result = client.audio.transcriptions.create(model=STT_MODEL, file=('speech.wav', audio), language='en')
                return result.text.strip()
            except Exception as e:
                self.logger.warning(f"OpenAI speech to text failed, trying Vosk: {e}")
        if self._vosk_available():
            from vosk import KaldiRecognizer
            if self._vosk_model is None:
                from vosk import Model, SetLogLevel
                SetLogLevel(-1)
                self._vosk_model = Model(VOSK_MODEL)
            recogniser = KaldiRecognizer(self._vosk_model, MIC_RATE)
            recogniser.AcceptWaveform(audio[44:])  # Skip the WAV header
            return json.loads(recogniser.FinalResult()).get('text', '').strip()
        return ''

    def _vosk_available(self):
        if not hasattr(self, '_has_vosk'):
            try:
                import importlib.util
                self._has_vosk = importlib.util.find_spec('vosk') is not None and os.path.isdir(VOSK_MODEL)
            except Exception:
                self._has_vosk = False
        return self._has_vosk

    def _record_sentence(self, mic, start_timeout):
        """
        Records one sentence: waits up to start_timeout seconds for speech to start, then records until
        silence_seconds of quiet (or max_record_seconds).

        Returns:
            bytes: The samples (with a little audio from just before the speech started), or b'' if none.
        """
        chunk_time = CHUNK / MIC_RATE
        before = deque(maxlen=int(0.3 / chunk_time))  # Keeps the start of the first word
        # Learn how loud the room is from the first 0.3 s
        noise = [chunk_level(mic.read()) for _ in range(10)]
        threshold = max(600.0, 2.5 * float(np.median(noise)))
        recorded, quiet_time, started = [], 0.0, None
        deadline = time.monotonic() + start_timeout
        while self.listening or self._listen_thread is None:
            chunk = mic.read()
            if not chunk:
                break
            self.sound_level = level = chunk_level(chunk)
            if started is None:
                before.append(chunk)
                if level > threshold:
                    started = time.monotonic()
                    recorded.extend(before)
                elif time.monotonic() > deadline:
                    return b''
                continue
            recorded.append(chunk)
            quiet_time = quiet_time + chunk_time if level < threshold else 0.0
            if quiet_time >= self.silence_seconds or time.monotonic() - started > self.max_record_seconds:
                break
        # Ignore clicks and bumps: real speech lasts longer than about 0.4 s
        return b''.join(recorded) if len(recorded) * chunk_time > self.silence_seconds + 0.4 else b''

    #-------------------------------------------
    # Listening on a background thread
    def start_listening(self, wake_word=True, respond=True, on_wake=None, on_heard=None, on_loud_sound=None,
                        on_chip_command=None):
        """
        Starts listening in a background thread. With the wake word, say "Hello Hiwonder", wait for "I'm here",
        then say a command. Each thing heard is understood, carried out and answered out loud (unless respond=False).

        Args:
            wake_word (bool, optional): True waits for "Hello Hiwonder" first; False responds to any speech.
            respond (bool, optional): True carries out commands and speaks replies; False only reports the words
                                      (to on_heard and last_heard), for your own code to act on.
            on_wake (function, optional): Called with no arguments when the wake word is heard.
            on_heard (function, optional): Called with the words heard, e.g. on_heard("move forward").
            on_loud_sound (function, optional): Called with the loudness when something louder than loud_threshold
                                                (a clap, a bang) is heard while waiting.
            on_chip_command (function, optional): Called with the code, e.g. (0x01, 0x00), of a command word
                                                  recognised by the WonderEcho chip itself (works offline).
        """
        if self.listening:
            return
        self.wake_word, self.respond = wake_word, respond
        self.on_wake, self.on_heard = on_wake or self.on_wake, on_heard or self.on_heard
        self.on_loud_sound = on_loud_sound or self.on_loud_sound
        self.on_chip_command = on_chip_command or self.on_chip_command
        if wake_word:
            self._open_serial()
        self.listening = True
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True, name='voice-listen')
        self._listen_thread.start()
        if self.describe_backends()['speech_to_text'] == 'none':
            self.logger.warning("No speech to text available (set an OpenAI API key or install Vosk), so only the "
                                "wake word, chip command words and loud sounds are reported.")

    def stop_listening(self):
        """Stops the listening thread."""
        self.listening = False
        if self._listen_thread is not None:
            self._listen_thread.join(timeout=3)
            self._listen_thread = None
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def stop(self):
        """Stops listening, speaking and any sound. Call when finished."""
        self.stop_listening()
        self.stop_speaking()

    def _open_serial(self):
        import serial
        try:
            self._serial = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0)
            self._serial.reset_input_buffer()
        except Exception as e:
            self._serial = None
            self.logger.warning(f"Can't open the WonderEcho serial port {SERIAL_PORT} ({e}); "
                                f"listening without the wake word")
            self.wake_word = False

    def _read_chip(self):
        """Reads the WonderEcho's serial port. Returns the (byte3, byte4) code of a whole packet, or None."""
        if self._serial is None:
            return None
        data = self._serial.read(self._serial.in_waiting or 1)
        self._chip_buffer = getattr(self, '_chip_buffer', b'') + data
        start = self._chip_buffer.find(b'\xaa\x55')
        if start < 0:
            self._chip_buffer = self._chip_buffer[-1:]
            return None
        packet = self._chip_buffer[start:start + 5]
        if len(packet) < 5:
            self._chip_buffer = self._chip_buffer[start:]
            return None
        self._chip_buffer = self._chip_buffer[start + 5:]
        return (packet[2], packet[3]) if packet[4] == 0xFB else None

    def _listen_loop(self):
        mic = MicStream()
        try:
            while self.listening:
                chunk = mic.read()
                if not chunk:
                    self.logger.warning("The microphone stopped")
                    break
                self.sound_level = level = chunk_level(chunk)
                if level > self.loud_threshold and self.on_loud_sound and not self.is_speaking() and not self.is_playing():
                    self._call(self.on_loud_sound, level)

                if self.wake_word:
                    code = self._read_chip()
                    if code is None or code == SLEEP_PACKET:
                        continue
                    if code != WAKE_PACKET:
                        self._chip_command(code)
                        continue
                    self.logger.info("wake word heard")
                    self._call(self.on_wake)
                    if os.path.exists(WAKE_SOUND):
                        self.play_sound(WAKE_SOUND, block=True)  # "I'm here"
                    mic.flush()
                    audio = self._record_sentence(mic, self.start_timeout)
                elif level > 600 and not self.is_speaking() and not self.is_playing():
                    audio = self._record_sentence(mic, 0.5)
                else:
                    continue

                if self.listening:
                    self._handle_audio(audio)
                # Don't hear our own voice: wait until the reply has been spoken, then throw that audio away
                while self.listening and (self.is_speaking() or self.is_playing()):
                    time.sleep(0.05)
                mic.flush()
        except Exception as e:
            self.logger.error(f"Listening stopped: {e}")
        finally:
            mic.close()
            self.listening = False

    def _chip_command(self, code):
        self.last_chip_command = code
        name = CHIP_COMMANDS.get(code)
        self.logger.info(f"Chip command word {code[0]:02X} {code[1]:02X}" + (f" ({name})" if name else ""))
        self._call(self.on_chip_command, code)
        if name and self.respond and name in self.commands:
            self.run_command(name)

    def _handle_audio(self, audio):
        if not audio:
            if self.wake_word and os.path.exists(NO_VOICE_SOUND):
                self.play_sound(NO_VOICE_SOUND, block=True)
            return
        if os.path.exists(HEARD_SOUND):
            self.play_sound(HEARD_SOUND)
        text = self.speech_to_text(audio)
        if not text:
            if self.describe_backends()['speech_to_text'] == 'none':
                self.say("I can hear you, but I need an API key or Vosk to understand words.")
            return
        print(f"Voice agent heard: {text}")
        self.logger.info(f"heard: {text}")
        self.last_heard = text
        self._call(self.on_heard, text)
        if self.respond:
            self.process_text(text, speak=True)

    def _call(self, callback, *args):
        """Runs a user callback without letting its errors stop the listening thread."""
        if callback:
            try:
                callback(*args)
            except Exception as e:
                self.logger.error(f"Callback {getattr(callback, '__name__', callback)} failed: {e}")

    #-------------------------------------------
    # Understanding and carrying out commands
    def process_text(self, text, speak=True):
        """
        Understands a sentence, carries out any commands in it and works out a reply. Used by the listening thread,
        and handy for testing without talking: VOICE.process_text("turn left 90 degrees").

        Args:
            text (str): What the user said.
            speak (bool, optional): True says the reply out loud.

        Returns:
            str: The reply.
        """
        with self._brain_lock:
            actions = []
            client, model = self._chat_client()
            reply = None
            if client:
                try:
                    reply = self._llm_reply(client, model, text, actions)
                except Exception as e:
                    self.logger.warning(f"language model failed, using keyword commands: {e}")
            if reply is None:
                reply = self._keyword_reply(text, actions)
            self.last_reply = reply
            self.history.append({'time': datetime.now().strftime('%H:%M:%S'), 'heard': text,
                                 'reply': reply, 'actions': actions})
            self.logger.info(f"reply: {reply}  actions: {actions}")
        if speak:
            self.say(reply)
        return reply

    def _llm_reply(self, client, model, text, actions):
        """Hiwonder's agent_process + llm_control loop: the model picks tools, we run them, it answers."""
        tools = [{'type': 'function', 'function': {'name': name, 'description': c['description'],
                                                   'parameters': c['parameters']}}
                 for name, c in self.commands.items()]
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT.format(name=self.name)}]
        messages += list(self._conversation)
        messages.append({'role': 'user', 'content': text})
        for _ in range(5):  # A few rounds of tool calls at most
            response = client.chat.completions.create(model=model, messages=messages, tools=tools, temperature=0.3)
            message = response.choices[0].message
            if not message.tool_calls:
                reply = re.sub(r'<think>.*?</think>', '', message.content or '', flags=re.DOTALL).strip()
                self._conversation.extend([{'role': 'user', 'content': text},
                                           {'role': 'assistant', 'content': reply}])
                return reply or "Done."
            messages.append({'role': 'assistant', 'content': message.content or '',
                             'tool_calls': [t.model_dump() for t in message.tool_calls]})
            for call in message.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments or '{}')
                except json.JSONDecodeError:
                    arguments = {}
                result = self.run_command(call.function.name, **arguments)
                actions.append({'command': call.function.name, 'arguments': arguments, 'result': result})
                messages.append({'role': 'tool', 'tool_call_id': call.id, 'content': str(result)})
        return "Sorry, that was too complicated for me."

    def _keyword_reply(self, text, actions):
        """The offline backup: recognises simple commands by their words (see _match_keywords)."""
        matches = self._match_keywords(text.lower())
        if not matches:
            return "Sorry, I don't know how to do that yet."
        results = []
        for name, arguments in matches:
            result = self.run_command(name, **arguments)
            actions.append({'command': name, 'arguments': arguments, 'result': result})
            results.append(str(result))
        return ' '.join(results)

    def _match_keywords(self, text):
        """
        Simple commands understood without a language model, e.g. "move forward two seconds", "turn left 90 degrees",
        "stop", "open the claw", "pick up", "what's the time". Commands added with add_command(phrases=[...]) are
        matched too.

        Returns:
            list: [(command name, arguments), ...]
        """
        for name, command in self.commands.items():
            if any(phrase in text for phrase in command.get('phrases', [])):
                return [(name, {})]
        seconds = parse_number(re.search(r'([\w.]+(?: and a half)?)\s+seconds?', text).group(1), None) \
            if re.search(r'[\w.]+\s+seconds?', text) else None
        metres = None
        found = re.search(r'([\w.]+(?: \w+)?)\s+(?:met(?:re|er)s?|m\b)', text)
        if found:
            metres = parse_number(found.group(1))
        found = re.search(r'([\w.]+(?: \w+)?)\s+(?:centimet(?:re|er)s?|cm\b)', text)
        if found and parse_number(found.group(1)) is not None:
            metres = parse_number(found.group(1)) / 100.0
        degrees = None
        found = re.search(r'([\w.]+(?: and \w+| \w+ and \w+| \w+)?)\s+degrees?', text)
        if found:
            degrees = parse_number(found.group(1))

        if re.search(r'\b(stop|halt|freeze|stay)\b', text):
            return [('stop', {})]
        if re.search(r'\b(spin|turn) (all the way )?around\b|full (turn|circle)|360', text):
            return [('turn', {'degrees': 360 if re.search(r'spin|full|360', text) else 180})]
        if re.search(r'\bturn\b|\brotate\b', text) and re.search(r'\b(left|right)\b', text):
            sign = 1 if 'left' in text else -1
            return [('turn', {'degrees': sign * (degrees or 90)})]
        for direction, words in (('forward', r'forwards?|ahead|straight'), ('backward', r'backwards?|back|reverse'),
                                 ('left', r'(slide|strafe|move|go) left'), ('right', r'(slide|strafe|move|go) right')):
            if re.search(rf'\b({words})\b', text):
                return [('move', {'direction': direction, 'seconds': seconds, 'distance': metres})]
        if re.search(r'\bpick (it |that |this )?up\b|\bgrab\b', text):
            return [('pick_up', {})]
        if re.search(r'\b(put|place|drop) (it |that |this )?down\b|\bdrop it\b|\blet go\b', text):
            return [('put_down', {})]
        for part, actions in (('claw', r'claw|gripper|hand'), ('arm', r'arm|lift')):
            if re.search(rf'\b({actions})\b', text):
                for action, words in (('open', r'open'), ('close', r'close|shut'), ('raise', r'raise|lift|up'),
                                      ('lower', r'lower|down')):
                    if re.search(rf'\b({words})\b', text):
                        if part == 'claw' and action in ('open', 'close'):
                            return [('claw', {'action': action})]
                        if action in ('raise', 'lower'):
                            return [('arm', {'action': action})]
        for emotion in ('happy', 'sad', 'cute', 'proud', 'playful', 'evil', 'wink', 'crying'):
            if re.search(rf'\b{emotion}\b', text) and re.search(r'\b(be|look|face|show|feel|make)\b', text):
                return [('show_emotion', {'emotion': emotion})]
        if re.search(r'\b(battery|voltage|charge)\b', text):
            return [('battery', {})]
        if re.search(r'\bwhat time\b|\bthe time\b', text):
            return [('tell_time', {})]
        if re.search(r'\b(hello|hi|hey)\b', text) or re.search(r'your name', text):
            return [('greet', {})]
        if re.search(r'\bwhat (can|do) you see\b|\blook at\b|\bdescribe\b', text):
            return [('describe_view', {'question': text})]
        return []

    def add_command(self, name, function, description, parameters=None, phrases=None):
        """
        Teaches the agent a new command. A language model calls it when the user asks for it (using the
        description); without one it is run when any of the phrases is heard.

        Args:
            name (str): Command name, e.g. 'dance'.
            function (function): Called with the parameters as keyword arguments. Whatever it returns is used in
                                 the reply (a string like "I danced" works well).
            description (str): What it does, for the language model, e.g. "Makes the robot dance".
            parameters (dict, optional): JSON schema properties, e.g. {'seconds': {'type': 'number'}}.
            phrases (list, optional): Words that trigger it without a language model, e.g. ['dance', 'boogie'].

        Example:
            VOICE.add_command('dance', my_dance, "Makes the robot dance", phrases=['dance'])
        """
        self.commands[name] = {
            'function': function, 'description': description,
            'parameters': {'type': 'object', 'properties': parameters or {}, 'required': []},
            'phrases': [p.lower() for p in (phrases or [])],
        }

    def remove_command(self, name):
        """Forgets a command (including the built-in ones, e.g. remove_command('move'))."""
        self.commands.pop(name, None)

    def run_command(self, name, **arguments):
        """
        Runs a command by name, e.g. run_command('turn', degrees=90).

        Returns:
            The command's result (usually a short sentence), or an error message.
        """
        command = self.commands.get(name)
        if command is None:
            return f"I don't have a command called {name}."
        try:
            result = command['function'](**arguments)
            self.logger.info(f"Ran {name}({arguments}) -> {result}")
            return result if result is not None else "Done."
        except Exception as e:
            self.logger.error(f"{name} failed: {e}")
            return f"Sorry, {name.replace('_', ' ')} didn't work: {e}"

    #-------------------------------------------
    # Built-in commands (ported from Hiwonder's llm_control tools, using the ClawPi interface)
    def _add_default_commands(self):
        number, string = {'type': 'number'}, {'type': 'string'}
        self.add_command('move', self._cmd_move,
            "Drive in a straight line. Give seconds or distance (metres); defaults to 2 seconds, at most 5.",
            {'direction': dict(string, enum=['forward', 'backward', 'left', 'right']),
             'seconds': number, 'distance': dict(number, description='metres')})
        self.add_command('turn', self._cmd_turn, "Turn on the spot. Positive degrees turn left, negative right.",
                         {'degrees': number})
        self.add_command('stop', self._cmd_stop, "Stop moving.")
        self.add_command('claw', self._cmd_claw, "Open or close the claw (gripper).",
                         {'action': dict(string, enum=['open', 'close'])})
        self.add_command('arm', self._cmd_arm, "Raise or lower the claw's arm.",
                         {'action': dict(string, enum=['raise', 'lower'])})
        self.add_command('pick_up', self._cmd_pick_up, "Pick up the object in front of the claw.")
        self.add_command('put_down', self._cmd_put_down, "Put down whatever the claw is holding.")
        self.add_command('show_emotion', self._cmd_emotion, "Show a face on the robot's screen.",
                         {'emotion': dict(string, enum=['happy', 'sad', 'wink', 'proud', 'evil', 'playful',
                                                        'cute', 'crying', 'awkward', 'dissatisfied'])})
        self.add_command('battery', self._cmd_battery, "Check the battery voltage.")
        self.add_command('describe_view', self._cmd_describe_view,
            "Look through the camera and answer a question about what the robot can see.",
            {'question': string})
        self.add_command('tell_time', lambda: f"It's {datetime.now().strftime('%-I:%M %p')}.",
                         "Say the current time.")
        self.add_command('greet', lambda: f"Hello, I'm {self.name}.", "Say hello and the robot's name.")

    def _need_robot(self):
        if self.robot is None:
            raise RuntimeError("no robot is connected")
        return self.robot

    def _cmd_move(self, direction='forward', seconds=None, distance=None):
        robot = self._need_robot()
        speed = getattr(robot, 'default_speed', 0.2)
        if distance:
            seconds = abs(distance) / speed
        seconds = min(float(seconds or 2.0), 5.0)  # Safety limit for spoken commands
        if direction in ('forward', 'backward'):
            robot.move_forward(seconds, direction=direction)
        else:
            robot.move_sideways(seconds, direction=direction)
        return f"I moved {direction} for {seconds:g} seconds."

    def _cmd_turn(self, degrees=90):
        robot = self._need_robot()
        degrees = max(-360.0, min(360.0, float(degrees)))
        try:
            robot.rotate_degrees(degrees)
        except RuntimeError:  # No odometry: turn for the time it should take
            speed = getattr(robot, 'default_turn_speed', 1.0)
            robot.rotate(abs(math.radians(degrees)) / speed, 'anticlockwise' if degrees > 0 else 'clockwise')
        return f"I turned {'left' if degrees > 0 else 'right'} {abs(degrees):g} degrees."

    def _cmd_stop(self):
        robot = self._need_robot()
        if hasattr(robot, 'stop_routine'):
            robot.stop_routine()
        robot.stop()
        return "Stopped."

    def _cmd_claw(self, action='open'):
        robot = self._need_robot()
        robot.open_claw() if action == 'open' else robot.close_claw()
        return f"Claw {'opened' if action == 'open' else 'closed'}."

    def _cmd_arm(self, action='raise'):
        robot = self._need_robot()
        robot.lift_claw() if action == 'raise' else robot.lower_claw()
        return f"Arm {'raised' if action == 'raise' else 'lowered'}."

    def _cmd_pick_up(self):
        self._need_robot().pick_up()
        return "I picked it up."

    def _cmd_put_down(self):
        self._need_robot().put_down()
        return "I put it down."

    def _cmd_emotion(self, emotion='happy'):
        self._need_robot().show_emotion(emotion)
        return f"Showing a {emotion} face."

    def _cmd_battery(self):
        volts = self._need_robot().get_battery_voltage()
        return f"My battery is at {volts:.1f} volts." if volts else "I couldn't read the battery."

    def _cmd_describe_view(self, question="What can you see?"):
        """Hiwonder's describe_current_view: sends the camera picture and the question to a vision model."""
        camera = getattr(self._need_robot(), 'CAMERA', None)
        frame = camera.get_rgb_frame() if camera else None
        if frame is None:
            return "I can't see anything, my camera isn't running."
        client = self._openai()
        if client is None:
            return "I need an OpenAI API key to describe what I see."
        import cv2
        ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image = 'data:image/jpeg;base64,' + base64.b64encode(jpeg.tobytes()).decode()
        response = client.chat.completions.create(model=VISION_MODEL, max_tokens=120, messages=[
            {'role': 'system', 'content': "You are a robot's eyes. Answer in one or two short spoken sentences."},
            {'role': 'user', 'content': [{'type': 'text', 'text': question},
                                         {'type': 'image_url', 'image_url': {'url': image}}]}])
        return response.choices[0].message.content.strip()

    #-------------------------------------------
    def get_status(self):
        """
        Everything the dashboard might show.

        Returns:
            dict: {'listening', 'speaking', 'sound_level', 'backends', 'last_heard', 'last_reply', 'history'}
        """
        return {'listening': self.listening, 'speaking': self.is_speaking(), 'sound_level': round(self.sound_level),
                'backends': self.describe_backends(), 'last_heard': self.last_heard, 'last_reply': self.last_reply,
                'history': list(self.history)}


#---------------------------------------------
# Test the voice agent on its own (no robot): python3 interfaces/voiceagentinterface.py
if __name__ == "__main__":
    VOICE = VoiceAgentInterface()
    print("Engines:", VOICE.describe_backends())
    VOICE.say("Hello, I am ready.", block=True)
    print('Typed test (no microphone):', VOICE.process_text("what time is it", speak=False))
    VOICE.start_listening(on_wake=lambda: print("Wake word!"),
                          on_heard=lambda text: print("Heard:", text),
                          on_loud_sound=lambda level: print("Loud sound", int(level)))
    try:
        input('Say "Hello Hiwonder" then a command. Press Enter to stop.\n')
    finally:
        VOICE.stop()
