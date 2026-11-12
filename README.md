# Primed

Primed measures how a movement changes across reps, sets, and days. An athlete picks an activity, uploads a phone clip, reviews the reps, and sets one as the baseline. A coach can approve that baseline and review the same measurements. Both use this one workflow.

The opening screen is a movement library, the clip in the centre, and the baseline on the right. Measurements are 2D projected angles and raw times from MediaPipe Pose Landmarker and OpenCV. They are not 3D joint angles, and a missing joint stays missing.

This repository did not contain an existing pipeline, notebook, pose model, or recording. The pipeline here is the first one. No example video is shipped. Upload a side-view clip, or load the labelled synthetic fixture when you only want to inspect the charts.

## Demo sequence

1. Install dependencies and the pose model (below).
2. Run `streamlit run app.py` and open the URL it prints.
3. The playground opens on Cricket bowling. **Open example** stays disabled because no recording is included. Read **What this movement can measure** for the candidate release and the measurements that are available.
4. Choose **Squat**. Upload a side-view MP4, MOV, M4V, or WebM under 24 MB and 45 seconds. Confirm the variant, camera view, and facing, then choose **Analyse clip**. The filename and any error stay on the page.
5. Use the frame slider for an exact analysed frame. The video player underneath seeks in whole seconds and is labelled that way. Knee angle, hip angle, and trunk tilt show **Unavailable** where the joints were not visible.
6. Select a rep and choose **Set as baseline**. That writes version 1. Setting another rep writes version 2 and keeps version 1. **Use** on an older version makes it active again without changing its numbers. **Approve reference** records a coach approval on the same id.
7. Upload a second squat from the same variant and view. Open **review** to see the selected rep as a solid line and the reference as a dashed line. Phase progress is for shape. Times use the clip timestamps. Correct a boundary or confirm an event under **Correct this rep**, and give the rep a set label.
8. Open **Progress**. All squat metrics are on one page. Switch **Reps**, **Sets**, and **Days**, and switch **Raw difference** and **Standardised (σ)**. Days use the workspace timezone. Sets use the labels you saved. The 2σ and 3σ lines are guides for the reference spread, not a confidence interval or a coaching judgement. A single reference rep supports raw differences and does not show sigma.
9. A deadlift baseline is not applied to a squat, and a squat baseline is not applied to a deadlift.
10. In **Sessions**, export the workspace. The JSON has measurements and reference versions, not the video. Import it in a new browser session and reattach a file with the same hash to see frames again.
11. Ask **Which rep changed the most?** The answer is computed from the stored measurements when `GEMINI_API_KEY` is unset.

To preview charts without a clip, choose **Preview charts with a labelled synthetic fixture**. The banner says the figures are not from a recording.

## Local setup

Python 3.12. FFmpeg and FFprobe must be on `PATH`. MediaPipe 0.10.35 is pinned to the `opencv-contrib-python` wheel it depends on. Do not also install `opencv-python` or `opencv-python-headless`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/fetch_pose_model.py
streamlit run app.py
```

`scripts/fetch_pose_model.py` downloads the official Pose Landmarker lite task file and checks its sha256:

`59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`

The model URL is the versioned asset `pose_landmarker_lite/float16/1`, not the floating `latest` alias. MediaPipe's task binary also needs `libEGL` (`libegl1` on Debian). It runs with the XNNPACK CPU delegate and a software GL context. No GPU is required.

Useful limits, all overridable with environment variables:

| Limit | Default | Variable |
| --- | --- | --- |
| Upload | 24 MB | `PRIMED_MAX_BYTES` |
| Duration | 45 s | `PRIMED_MAX_DURATION_S` |
| Analysed frames | 480 | `PRIMED_MAX_FRAMES` |
| Sample rate | 12 fps, lowered to fit the frame budget | `PRIMED_SAMPLE_FPS` |
| Frame width | 960 px | `PRIMED_MAX_WIDTH` |
| Reference reps before sigma | 5 | `PRIMED_MIN_REFERENCE_REPS` |

Sigma is also withheld when the reference spread is effectively zero. One reference rep supports raw differences only.

## What is measured

Profiles live in `primed/profiles.py`. Adding a movement there is enough for the library, review, and progress screens.

| Movement | Detection | Notes |
| --- | --- | --- |
| Squat | Knee-angle dip | Side view. Bottom knee and hip angles, hip-drop proxy, trunk tilt, descent and ascent time. |
| Deadlift | Hip-angle rise | Lift-off and knee pass stay candidates because the bar is not tracked. |
| Bench press | Elbow-angle dip | Bar path is unavailable. |
| Overhead press | Elbow-angle rise | |
| Reverse lunge | Front-leg knee dip | Set the front leg. Otherwise the leg is a guess and the rep is uncertain. |
| Pull-up | Elbow-angle dip | Chin over the bar is not detected. |
| Kettlebell swing | Hip-angle rise | Cadence is 60 divided by the cycle duration. The bell is not tracked. |
| Golf swing | Manual | Impact stays a candidate until you confirm it. |
| Cricket bowling | Candidate window | Elbow, knee, and trunk are measured. Release is the fastest 2D wrist movement of the bowling arm, not a detected ball. |
| Tennis serve | Manual | Contact is a candidate. |
| Free throw | Manual | Release is a candidate. |
| Baseball pitch | Manual | Release is a candidate. |

Angles use pixel coordinates after converting normalised landmarks with the frame width and height, so x and y share a unit. The angle at joint B is between vectors A−B and C−B. A zero-length vector is missing. The cosine is clamped before `arccos`.

Trunk tilt is a side-view angle. Positive means forward lean toward the facing direction you select (right means the face points to the right of the frame). Front view does not invent a sign.

Timing metrics use the clip's own timestamps. Phase alignment is only for the shape overlay and the dashed reference trace. A gap of 0.2 seconds or less may be bridged on a plotted line. Longer gaps stay empty. Scalar metrics use the raw samples.

Automatic thresholds are documented on each profile and in `SEG_VERSION`. Manual edits are stored with the source timestamps and are kept when segmentation is run again.

## Baselines and drift

A reference belongs to one workspace, movement, variant, and camera view. Handedness and load are recorded. Replacing a baseline creates a new version. The previous version stays readable.

The scalar reference is the mean of the valid measurements. The trace is the pointwise mean on equal-width phase progress. Signed change is observed minus reference, in the metric's own unit. Standardised drift is the absolute signed change divided by the sample standard deviation of the approved reference reps.

Reps show each rep's change. Sets and days show the median signed change, or the median absolute standardised deviation on the sigma chart. A set is a label you confirmed. A day is a calendar day in the workspace timezone. A group with no qualifying reps is a gap.

## Ask Primed

With no API key, Ask Primed answers from the stored measurements. With `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) it calls the Gemini API through `google-genai`. The model id defaults to `gemini-3.8-flash` and can be changed with `PRIMED_GEMINI_MODEL`. Only derived measurements and the question are sent. Cited ids, units, values, and timestamps are checked against those records. A mismatch is replaced with the measured answer. The key is not written to the workspace, the browser, or the logs.

Gemini Live voice review is not part of this build.

## Sessions

Each browser session has its own folder under `~/.local/share/primed/sessions`, or `PRIMED_DATA_DIR`. Ordinary Streamlit reruns keep the session. Restarting the process or opening a new browser starts a new folder. Export and import are how an athlete continues across days. This is not a shared login.

`PRIMED_ENABLE_CLOUD=1` does not turn on Cloud Storage or Firestore. The public demo refuses that flag until a project, a private bucket, and access rules exist. Do not set it for the first deployment.

## Docker and Cloud Run

The Docker daemon was not available in the environment that built this app, so the image has not been run here. The Dockerfile uses Python 3.12, FFmpeg, and the EGL libraries MediaPipe needs, listens on `0.0.0.0` and the injected `PORT`, and downloads the pose model during the build.

```bash
docker build -t primed .
docker run --rm -p 8080:8080 -e PORT=8080 primed
```

A public demo, after billing, the Cloud Run API, and the Cloud Build API are enabled on your project:

```bash
gcloud run deploy primed --source . --region asia-south1 \
  --allow-unauthenticated --cpu 2 --memory 4Gi
```

Pass the Gemini key as a Cloud Run environment variable or secret named `GEMINI_API_KEY`. Do not bake it into the image. Optional flags that help a Streamlit session are `--session-affinity` and a request timeout long enough for a short clip. The upload cap is 24 MB so a multipart request stays under Cloud Run's 32 MiB HTTP/1 limit.

After deployment, upload a short side-view clip from the playground and confirm the filename, a frame preview, and either measured angles or an explicit tracking-gap message. Raw videos and workspace exports stay out of the image.

Session files on Cloud Run are ephemeral. Export the workspace to keep it.

## Tests

```bash
pytest
```

Covered here: projected angles and trunk-tilt sign, squat and deadlift segmentation, manual boundaries kept across re-segmentation, cricket release left as a candidate, baseline versions, sigma withheld for a single rep and for zero spread, set medians, timezone day grouping, a squat reference refused for a deadlift, export and import, and Ask Primed with a bad id. A generated clip with no person is run through MediaPipe and produces an empty pose, not invented angles.

## Verification in this environment

- Python 3.12.3, FFmpeg 6.1.1, MediaPipe 0.10.35, OpenCV 4.10.0 (contrib wheel only), Streamlit 1.51.0, Plotly 6.4.0, `google-genai` 1.47.0.
- Pose Landmarker lite loaded and returned no landmarks for a blank frame and for a generated clip with no person.
- No cricket, squat, or deadlift recording was in the repository, so none of those movements were verified on real footage.
- No `GEMINI_API_KEY` was available, so a live Gemini call was not made. The no-key path and the id check were tested.
- Docker was not installed, so the image was not built.

## Layout

- `app.py` — `streamlit run app.py`
- `primed/` — profiles, video, pose, metrics, baselines, workspace, Ask Primed, screens
- `scripts/fetch_pose_model.py` — official model download
- `tests/` — measurement and workflow tests
- `models/pose_landmarker_lite.task` — downloaded, not committed
- Fonts in `primed/static/fonts/` are SIL Open Font License files from Google Fonts (Bricolage Grotesque and Instrument Sans). The license is `primed/static/fonts/OFL.txt`.
