# SmartCam: a CCTV that understands what it sees

SmartCam watches a camera and tells you what's happening: who arrived, how long they stayed, who walked into an area you marked, who is hanging around, when a crowd forms, and when a bag is left alone. Every event is saved with a snapshot. Your free Gemini key can also describe the scene in plain English, answer questions about it ("Is the door open?"), and summarise the day.

Detection runs **on this laptop**: no cloud, no GPU, about one CPU core while people are moving and much less when the scene is still. Only the optional AI descriptions go online.

## Start and stop

- **Start:** double-click the **SmartCam** icon on the desktop (or `run.bat`). The dashboard opens in its own window at http://127.0.0.1:47920. SmartCam doesn't start by itself when Windows starts. The icon is drawn by `make_icon.py`.
- **Try it without a camera:** `run.bat --camera demo` plays a street scene clip. Or choose *Demo video* in Settings.
- **Stop:** press the power button (top right). Closing the window alone keeps SmartCam watching in the background, and running `run.bat` again reopens the window.

## First time on a new computer

1. Run `setup.bat`. It builds SmartCam's own Python environment in `venv`.
2. Get the object detector. It isn't in this repo because of its AGPL-3.0 licence. In any Python environment
   that isn't SmartCam's own, run `pip install ultralytics onnx` and then `yolo export model=yolo11n.pt format=onnx`.
   This downloads Ultralytics' official YOLO11n weights and converts them. Put the `yolo11n.onnx` it makes in a
   `models` folder next to `smartcam.py`.
3. Start it with `run.bat`. It makes `config.json` with the default settings.

## Cameras

Choose one in **Settings → Camera**:

| Camera | What to enter |
|---|---|
| Laptop webcam | *Built-in webcam* (`0`) |
| USB webcam | *Second camera* (`1`) |
| Phone as camera | Install **IP Webcam** (Android), press *Start server*, then enter `http://PHONE-IP:8080/video` |
| CCTV / IP camera | Its RTSP address, e.g. Hikvision `rtsp://user:pass@IP:554/Streaming/Channels/101`, Dahua `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0`, Tapo `rtsp://user:pass@IP:554/stream1`, Reolink `rtsp://user:pass@IP:554/h264Preview_01_main` |
| Recorded video | The full path to the file, e.g. `C:\Users\farha\Videos\door.mp4` |

If the camera drops out, SmartCam keeps retrying by itself.

## What it notices

| Event | When |
|---|---|
| **Visit** (person, vehicle, animal) | Something new comes into view. The row later shows how long it stayed. |
| **Zone alert** | A person, vehicle or animal walks into an area you drew (their feet count, not their head). |
| **Loitering** | A person stays longer than the set time (default 60 s). |
| **Crowd** | The set number of people (default 5) are in view at once. |
| **Unattended bag** | A backpack, handbag or suitcase sits still with nobody next to it (default 30 s). |

Alerts show in red, pop up at the top and beep (you can switch the beep off in Settings). It also recognises 80 everyday objects (chairs, laptops, phones and so on) and lists them under the picture, but only the things above create events.

**Zones:** press **Zones** and click the corners of an area on the picture. Click the first corner again to close the shape, then name it and save.

## AI descriptions (optional)

These use the same free `GEMINI_API_KEY` as JARVIS and ZACK, so they share its daily limit.

- **Automatic:** when something happens, one snapshot goes to Gemini and its description is added to the event. That's at most one every 45 s and 150 a day; both limits can be changed in Settings.
- **Describe now / Ask / Summarise today** run only when you press them. *Summarise* sends the day's event list as text only.
- **Privacy:** snapshots sent to Gemini leave your laptop, and Google's free tier may use them to improve its products. Turn off *Describe events automatically* if the camera sees anything private. Everything else stays on the laptop. Gemini is told never to guess anyone's identity, age or ethnicity.

## Where things are

| Path | What |
|---|---|
| `config.json` | Your settings and zones (the dashboard edits it for you) |
| `data/events.db` | The event log (SQLite) |
| `data/snapshots/<day>/` | Snapshot JPEGs, deleted after *Keep for* days (default 14) |
| `data/smartcam.log` | Diagnostics |
| `models/yolo11n.onnx` | The object detector (YOLO11n, converted from Ultralytics' official weights; not on GitHub, see *First time on a new computer*) |
| `samples/vtest.avi` | The demo clip (from OpenCV's sample data) |

## How it works

1. **Camera thread** reads frames and keeps only the newest.
2. **Motion check** compares tiny grey copies of each frame. If nothing moves, YOLO looks only every 1.5 s. When something moves, it looks 4 times a second (the *Looks per second* setting).
3. **YOLO11n on ONNX Runtime** finds objects (about 55–100 ms per look on this laptop's CPU).
4. **Tracker** gives each object an ID and follows it. It predicts where each object is heading, lets half-hidden sightings continue a track, and requires 2 sightings before anything counts (and a surer first sighting for animals and bags, so a rock doesn't become a dog).
5. **Rules** turn tracks into events. Events go to the log with a snapshot and, optionally, to Gemini.
6. **Dashboard** (`web/index.html`) shows the live picture as an MJPEG stream and checks for news every second.

The server only accepts connections from this laptop. It refuses requests that come from other websites or other hostnames, because the camera picture is private.

## Troubleshooting

- **"NO SIGNAL" with the webcam:** another app (Teams, Zoom, Camera) may be using it. Close that app; SmartCam reconnects by itself.
- **Window didn't open:** browse to http://127.0.0.1:47920.
- **Rebuild the Python environment:** delete `venv` and run `setup.bat`.

## Licence note

YOLO11 is AGPL-3.0 (Ultralytics). It's fine for personal use. Selling SmartCam, for example through Mavetech, would need an Ultralytics Enterprise licence or a detector with a permissive licence.
