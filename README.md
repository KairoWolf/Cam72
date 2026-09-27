# PuppyCam

Finds, counts and highlights newborn puppies on the cameras you already have, and warns your
phone when something needs you:

- **A puppy has left the pile.** One puppy away from mom and its littermates for over a minute (it can get cold).
- **Not all puppies are visible.** Fewer than 9 visible on every camera for 10 minutes. Check that no puppy is stuck under mom or outside the box.
- **Mom has left.** Mom gone for 20 minutes (only once you have labeled mom a few times).
- **A camera is offline.**

Every puppy gets its own colored outline and number on the live view. With two cameras, the
count comes from whichever camera sees more puppies, so a puppy hidden in one view is still
counted if the other view sees it, and no puppy is ever counted twice.

![Live view](docs/live-view.jpg)

*The screenshot above comes from a demo model trained on only 3 of your screenshots. It shows
what the highlighting looks like; it is not a measure of accuracy.*

## How accurate is it?

No camera system can promise 100%, because a puppy lying completely under mom can't be seen by
any camera. Everything else is built to get as close as possible:

| What it does | Why |
| --- | --- |
| Trains on **your** frames from **your** cameras, labeled by you | Off-the-shelf AI models find mom but miss almost all newborn puppies (we tested them on your screenshots) |
| Labeling is fast: drag a box, and the AI (SAM 2) traces the puppy's outline | More labeled frames means higher accuracy |
| After the first model, new frames come **pre-labeled**; you only fix mistakes | Each round of labeling gets faster |
| **Saves the frames the model is unsure about** for you to label | These teach the model the most |
| Tunes the confidence threshold for **exact counts** | It optimizes the number you care about, not a generic score |
| Measures accuracy on **held-out frames from other hours** and shows it | You see the real accuracy, not an optimistic one |
| A new model only replaces the current one if it counts at least as well | Retraining never makes it worse |
| Never reports more than 9; smooths the count over a few seconds; uses 2 camera views | Removes flicker, extra false detections, and hidden-puppy gaps |
| Trains and detects in grayscale | Night vision and daytime color look the same to the model |

Plan on a few labeling rounds. About 30 labeled frames gives a first useful model; 150 to 300+
frames from both cameras (day and night, feeding and sleeping, mom in and out) is where it gets
very reliable. The **Train** page shows the exact-count accuracy after every round.

## What you need

- A Windows or Linux PC with an NVIDIA GPU (any RTX card is plenty) and a current NVIDIA driver.
- **Docker Desktop** (Windows) or Docker Engine + NVIDIA Container Toolkit (Linux).
- Your cameras' RTSP addresses (see [Camera addresses](#camera-addresses)).

## Setup (Windows)

1. **Install or update the NVIDIA driver** from nvidia.com (GeForce Experience or the NVIDIA app works too).
2. **Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)** with the default
   WSL 2 option, restart, and start Docker Desktop once.
3. **Check that Docker can see the GPU.** Open PowerShell and run:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
   ```
   You should see your graphics card listed.
4. **Get this project**: on GitHub click **Code → Download ZIP** and unzip it, or run
   `git clone https://github.com/KairoWolf/Cam72.git`.
5. **Configure it.** In the project folder, copy `.env.example` to `.env` and open `.env` in Notepad:
   ```ini
   CAM1_NAME=Wyze top
   CAM1_URL=rtsp://user:password@192.168.1.50/live
   CAM2_NAME=Side cam
   CAM2_URL=rtsp://user:password@192.168.1.51:554/stream1
   EXPECTED_PUPPIES=9
   NTFY_TOPIC=puppies-choose-a-long-random-name-123
   TZ=America/New_York
   ```
6. **Start it** from PowerShell inside the project folder:
   ```powershell
   docker compose up -d --build
   ```
   The first start downloads about 5 GB, so give it a while.
7. **Open <http://localhost:8080>.** From your phone on the same Wi-Fi, use `http://<your-PC-IP>:8080`.
   To find the PC's address, run `ipconfig` and look for the IPv4 address.

To stop PuppyCam, run `docker compose down`. Your labels and models stay in the `data` folder.

On Linux, install Docker Engine and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then follow steps 3 to 7.

Without an NVIDIA GPU, it still runs, just slowly:
`docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build`.

## Camera addresses

Test any address in VLC first (**Media → Open Network Stream**). If VLC plays it, PuppyCam can use it.

- **Wyze Cam v2, v3, or Pan v1 with Wyze's RTSP firmware:** in the Wyze app, go to
  **Settings → Advanced Settings → RTSP**. The app shows the URL, for example `rtsp://user:pass@192.168.1.50/live`.
- **Any other Wyze camera (v4, Pan v3, OG, ...) through the included bridge.** No firmware change is needed:
  1. Get an API ID and key at <https://developer-api-console.wyze.com/#/apikey/view> and put them in
     `.env` along with your Wyze email and password (`WYZE_EMAIL`, `WYZE_PASSWORD`, `WYZE_API_ID`, `WYZE_API_KEY`).
  2. Start with the bridge: `docker compose --profile wyze up -d --build`.
  3. Open <http://localhost:5080> to see your camera names, then set
     `CAM1_URL=rtsp://wyze-bridge:8554/<camera_name>` (lowercase, spaces become `_`, for example `puppy_cam`).
- **Tapo / TP-Link:** create a camera account in the Tapo app (**Camera Settings → Advanced Settings → Camera Account**), then use
  `rtsp://user:pass@IP:554/stream1`.
- **Reolink:** `rtsp://admin:pass@IP:554/h264Preview_01_main`. **Amcrest / Dahua:**
  `rtsp://admin:pass@IP:554/cam/realmonitor?channel=1&subtype=0`. For other brands, search for
  "*brand* RTSP URL".

Give the cameras fixed IP addresses in your router so the URLs keep working.

## Using it

### 1. Collect frames

PuppyCam saves frames for labeling automatically: one per camera every 10 minutes, plus frames
where the model is unsure (once a model exists). You can also:
- click **Save frame for labeling** on the live page when something interesting happens, or
- on the **Label** page, **upload** screenshots or videos. Wyze event clips and phone screenshots
  work; black bars around screenshots are removed automatically.

Aim for variety: night and day, puppies piled up and spread out, feeding, mom in and out of the box.

### 2. Label

On the **Label** page, drag a box around one puppy. The AI traces its outline and gives it a number.
Repeat for every puppy you can see, then press **Enter** to save and go to the next frame.

- Label every puppy that is at least partly visible (only its head, back or rump showing still counts).
  Box only the visible part.
- One puppy per box. If an outline covers two puppies, press **Ctrl+Z** and draw a tighter box, or
  press **B** to keep plain boxes.
- Press **2** to label mom on some frames. This enables the "mom left" alert and teaches the model
  that mom's paws are not puppies. Press **1** to switch back to puppies.
- Once a model exists, frames arrive pre-labeled with dashed outlines. Fix what's wrong, delete
  extras (click, then **Del**), click a dotted **?** outline to accept it, and save.
- If the frame shows fewer than 9 puppies, that's fine. Label what you see.
- Controls: mouse wheel zooms, right-drag pans, **R** resets the view, and **H** hides outlines so you can see the picture.

### 3. Train

On the **Train** page, click **Start training**. The defaults (Large model, 150 epochs, 1280 px)
take roughly 10 to 30 minutes on a modern GPU, depending on how many frames you have. When it
finishes, the page shows:

- **Exact count**: the share of held-out test frames where the model counted exactly the puppies you labeled.
- **Mistakes**: the test frames it got wrong, with links. Check each one. Sometimes the label is
  what's wrong. Then label more frames like it.

The live view switches to the new model automatically if it counts at least as well as the old one.

### 4. Repeat

Label the new frames (the uncertain ones come first in the queue), train again, and watch the
exact-count number climb. For later rounds, choosing **Continue from the model in use** as the model
size trains faster, because it starts from what the current model already knows.

## Phone alerts

1. Install **ntfy** ([Android](https://play.google.com/store/apps/details?id=io.heckel.ntfy) /
   [iPhone](https://apps.apple.com/app/ntfy/id1625396347)).
2. Subscribe to a topic with a long, random name, and put the same name in `.env` as `NTFY_TOPIC`.
3. Restart: `docker compose up -d`.

Alerts arrive with a snapshot of the highlighted camera view. Critical alerts (a puppy away from the
pile, puppies missing) are sent as urgent and repeat every 15 minutes while they last. You can also
tick **Alarm sound** on the live page.

`WEBHOOK_URL` additionally posts every alert as JSON to Home Assistant, Discord, Slack and similar services.

## Settings

All settings live in `.env`; restart after changing them (`docker compose up -d`).

| Setting | Default | Meaning |
| --- | --- | --- |
| `CAM1_URL` ... `CAM8_URL`, `CAM1_NAME` ... | - | Cameras (RTSP/HTTP URL, video file or image folder) |
| `EXPECTED_PUPPIES` | 9 | Litter size |
| `AWAY_SECONDS` | 60 | How long a puppy must be away from the pile before alerting |
| `AWAY_FACTOR` | 0.75 | How far is "away", in puppy lengths |
| `COUNT_LOW_MINUTES` | 10 | Alert when not all puppies have been visible for this long |
| `MOM_AWAY_MINUTES` | 20 | Alert when mom has been gone this long |
| `OFFLINE_MINUTES` | 2 | Alert when a camera sends no video for this long |
| `ALERT_REPEAT_MINUTES` | 15 | Repeat an ongoing alert this often |
| `NTFY_TOPIC`, `NTFY_SERVER`, `NTFY_TOKEN` | - | Phone notifications through ntfy |
| `WEBHOOK_URL` | - | Also send alerts to a webhook |
| `WEB_PASSWORD` | - | Require a password for the web page |
| `IMGSZ` | 1280 | Detection resolution. Higher finds small puppies better |
| `PROCESS_FPS` | 4 | Detection passes per second (all cameras together) |
| `CONF` | tuned | Override the confidence threshold chosen during training |
| `CLAMP_TO_EXPECTED` | true | Never report more puppies than the litter size |
| `GRAYSCALE` | true | Treat color (day) and infrared (night) frames alike |
| `CAPTURE_EVERY_MINUTES` | 10 | Save a frame for labeling this often (0 turns it off) |
| `CAPTURE_UNCERTAIN` | true | Save frames where the model is unsure |
| `DEVICE` | auto | `0` for the first GPU, `cpu` to force the CPU |
| `TZ` | UTC | Your time zone, for example `America/New_York` or `Europe/London` |

## Troubleshooting

- **`could not select device driver "nvidia"`**: Docker can't see the GPU. Update the NVIDIA driver and Docker
  Desktop and check step 3 of the setup. On Linux, install the NVIDIA Container Toolkit.
- **A camera stays "connecting"**: test the URL in VLC. Check the IP address, user name and password.
  If a password contains special characters such as `@`, replace them with their URL encoding (`@` becomes `%40`).
- **The Wyze bridge can't connect** (`IOTC_ER_TIMEOUT`, discovery timeouts): Docker's network can block the
  camera's replies. In Docker Desktop, go to **Settings → Resources → Network** and enable host networking.
  Add `network_mode: host` to the `wyze-bridge` service (and remove its `ports:`), then use
  `CAM1_URL=rtsp://host.docker.internal:8554/<camera_name>`.
- **Training runs out of GPU memory**: choose a smaller model size on the Train page, or set the image size to 960.
- **Counts jump around**: label more frames, especially the ones listed under **Mistakes** and the automatically saved
  "uncertain" frames, and train again.

## Good to know

- The puppy numbers (#1 to #9) are labels for the current view, not names. Each camera numbers its puppies separately,
  and a puppy that stays hidden for a while may come back with a different number.
- The count uses the camera that sees the most puppies. It never adds the two views together (that could count a puppy
  twice), so a puppy only counts if at least one camera sees it.
- Everything runs on your PC. Video never leaves your network. Only alert snapshots go out, if you enable ntfy or a webhook.
- Don't forward port 8080 on your router. For access away from home, use a VPN such as Tailscale, and set `WEB_PASSWORD`.

## How it works

```
cameras (RTSP) ──► readers (latest frame only) ──► YOLO26 segmentation model, all cameras in one GPU batch
                                                       │
       ┌───────────────────────────────────────────────┘
       ▼
 per camera: threshold + hysteresis ─► keep the 9 most confident ─► stable numbers (IoU tracker)
             ─► "away from the pile" (grouping by distance) ─► count smoothed over 5 s
       ▼
 fusion: best camera's count, mom seen anywhere ─► alert rules ─► ntfy / webhook
       ▼
 live view: colored outlines + numbers (MJPEG)       capture policy ─► frames to label
 labeling: box ─► SAM 2 outline ─► YOLO segmentation labels ─► training ─► threshold tuned for exact counts
```

The code is in `puppycam/`: `engine.py` runs the loop, `analysis.py` handles counting and the "away" logic,
`alerts.py` the alert rules, `training.py` training and evaluation, `server.py` and `web/` the web app.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -r requirements-dev.txt
pytest

# run against the sample screenshots instead of real cameras
CAM1_URL=tests/assets/wyze_cam_1.jpg CAM2_URL=tests/assets/side_cam_2.jpg DATA_DIR=./data python -m puppycam
```

To train from the command line instead of the web page:
`docker compose exec puppycam python -m puppycam.training --epochs 150`
(add `--help` for all options).

## Credits

- [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO26 for detection and training (AGPL-3.0), and its
  SAM 2 integration ([Meta's Segment Anything 2](https://github.com/facebookresearch/sam2)) for the labeling outlines.
- [docker-wyze-bridge](https://github.com/IDisposable/docker-wyze-bridge) (IDisposable's maintained fork of mrlt8's
  bridge) to get RTSP from Wyze cameras.
- [ntfy](https://ntfy.sh) for phone notifications.
