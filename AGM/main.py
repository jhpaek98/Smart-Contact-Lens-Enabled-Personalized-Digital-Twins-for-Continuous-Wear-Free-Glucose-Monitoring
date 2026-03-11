# cd "your path for "main.py""
# py -m uvicorn main:app --reload --reload-dir templates --reload-dir static

from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from typing import List
import pandas as pd
import os
import uuid
from PIL import Image
import json
from datetime import datetime
from difflib import get_close_matches
import matplotlib
import numpy as np
import re
import tempfile
import ffmpeg

from faster_whisper import WhisperModel

from ml_model import extract_feature
from ML_training import train_model_from_titles
from predict_from_title import predict_from_title
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ffmpeg_dir = os.getenv("FFMPEG_DIR")
if ffmpeg_dir and os.path.isdir(ffmpeg_dir):
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

client = OpenAI()

# Matplotlib settings
matplotlib.rcParams["axes.unicode_minus"] = False

client = OpenAI()

# App initialization
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# File path constants
DATA_DIR = "static/data"
UPLOAD_DIR = "static/uploads"
LABEL_FILE = os.path.join(DATA_DIR, "labels.json")
EVENT_FILE = os.path.join(DATA_DIR, "events.json")
CHAT_LOG_FILE = os.path.join(DATA_DIR, "chat_logs.json")
FEATURE_LABEL_FILE = os.path.join(DATA_DIR, "feature_labels.json")

# Create folders
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


# Whisper STT
_model = WhisperModel("base", device="cpu", compute_type="int8")


def to_wav_16k(in_path: str, out_path: str):
    try:
        (
            ffmpeg
            .input(in_path)
            .output(out_path, ac=1, ar=16000, f="wav")
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error:
        raise HTTPException(status_code=400, detail="FFmpeg conversion failed")


@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
        tmp_in.write(await file.read())
        in_path = tmp_in.name
    out_path = in_path + ".wav"

    try:
        to_wav_16k(in_path, out_path)
        segments, _ = _model.transcribe(out_path, language="en", vad_filter=True)
        text = "".join(s.text for s in segments).strip()
        return {"text": text}
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except Exception:
                pass


# Global state
uploaded_graphs = []
session_chat_history = []
chat_history = []

# Load existing data
if os.path.exists(LABEL_FILE):
    uploaded_graphs = json.load(open(LABEL_FILE, encoding="utf-8"))
if os.path.exists(CHAT_LOG_FILE):
    chat_history = json.load(open(CHAT_LOG_FILE, encoding="utf-8"))
if not os.path.exists(EVENT_FILE):
    json.dump({}, open(EVENT_FILE, "w", encoding="utf-8"))


def save_chat_log(log=None):
    data = log if log is not None else (
        json.load(open(CHAT_LOG_FILE, encoding="utf-8"))
        if os.path.exists(CHAT_LOG_FILE) else []
    )
    with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_title_from_message(message: str, uploaded_graphs: list) -> str:
    all_titles = [g["title"] for g in uploaded_graphs]
    for title in sorted(all_titles, key=len, reverse=True):
        if title in message:
            return title
    match = get_close_matches(message, all_titles, n=1, cutoff=0.5)
    return match[0] if match else None


@app.get("/list_images")
def list_images(dir: str = Query("", alias="dir")):
    base = Path("static") / "uploads" / dir.strip("/\\")
    if not base.exists() or not base.is_dir():
        return {"files": []}
    exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
    files = sorted(
        [p.name for p in base.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=str.lower
    )
    return {"files": files}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, tab: str = "chat"):
    events = json.load(open(EVENT_FILE, encoding="utf-8"))

    if tab == "history":
        with open(CHAT_LOG_FILE, encoding="utf-8") as f:
            full_chat_history = json.load(f)
    else:
        full_chat_history = []

    biomarker_subject_map = {}
    for g in uploaded_graphs:
        b = g["biomarker"]
        s = g["subject_id"]
        biomarker_subject_map.setdefault(b, set()).add(s)
    biomarker_subject_map = {k: list(v) for k, v in biomarker_subject_map.items()}

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "tab": tab,
        "uploaded_graphs": uploaded_graphs,
        "chat_history": full_chat_history if tab == "history" else sorted(session_chat_history, key=lambda x: x["timestamp"]),
        "events": events,
        "biomarker_map": biomarker_subject_map,
        "css_version": int(datetime.now().timestamp())
    })


@app.post("/uploadfile")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(...),
    subject_id: str = Form(...),
    biomarker: str = Form(...),
    task_type: str = Form(...),
    disease_status: str = Form(...)
):
    try:
        ext = ".xlsx" if file.filename.endswith(".xlsx") else ".csv"
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(DATA_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(await file.read())

        uploaded_graphs.append({
            "title": title,
            "filename": filename,
            "subject_id": subject_id,
            "biomarker": biomarker,
            "task_type": task_type,
            "disease_status": disease_status
        })

        json.dump(uploaded_graphs, open(LABEL_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return RedirectResponse(url="/?tab=record", status_code=303)

    except Exception as e:
        return HTMLResponse(content=f"<h1>Upload failed: {e}</h1>", status_code=500)


@app.post("/deletefile")
async def delete_file(request: Request, filename: str = Form(...)):
    global uploaded_graphs
    try:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            os.remove(filepath)

        old_graphs = uploaded_graphs[:]
        titles_to_remove = [g["title"] for g in old_graphs if g["filename"] == filename]

        uploaded_graphs = [g for g in uploaded_graphs if g["filename"] != filename]
        json.dump(uploaded_graphs, open(LABEL_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        events = json.load(open(EVENT_FILE, encoding="utf-8"))
        for title in titles_to_remove:
            if title in events:
                del events[title]
        json.dump(events, open(EVENT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        return RedirectResponse(url="/?tab=record", status_code=303)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Delete failed: {e}</h1>", status_code=500)


@app.get("/data/{data_filename}")
async def get_chart_data(data_filename: str):
    try:
        filepath = os.path.join(DATA_DIR, data_filename)
        ext = os.path.splitext(filepath)[1]
        df = pd.read_excel(filepath) if ext == ".xlsx" else pd.read_csv(filepath)

        y_label = df.columns[1]
        labels = df.iloc[:, 0].astype(str).tolist()
        values = df.iloc[:, 1].tolist()
        title = next((g["title"] for g in uploaded_graphs if g["filename"] == data_filename), None)
        events = json.load(open(EVENT_FILE, encoding="utf-8"))
        return {"labels": labels, "values": values, "label": y_label, "events": events.get(title, [])}
    except Exception as e:
        return {"error": str(e)}


@app.post("/chat")
async def chat(message: str = Form(...)):
    if re.search(r"(glucose scheduler|schedule|plan|scheduler)", message, re.IGNORECASE):
        return RedirectResponse(url="/?tab=scheduler", status_code=303)


@app.post("/chat", response_class=HTMLResponse)
async def chat_gpt(request: Request, message: str = Form(...)):
    global chat_history

    matched_title = extract_title_from_message(message, uploaded_graphs)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_entry = {"role": "user", "content": message, "timestamp": timestamp}
    session_chat_history.append(user_entry)

    try:
        task_type_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the following message as exactly one of: "
                        "'analysis request', 'add event', 'delete event', "
                        "'guideline', 'prediction', or 'other'."
                    )
                },
                {"role": "user", "content": message}
            ]
        )
        task_type = task_type_response.choices[0].message.content.lower()

        if "guideline" in task_type:
            reply = (
                "Which guideline would you like?\n"
                "- Event addition guideline\n"
                "- Graph download guideline\n"
                "- Feature extraction guideline\n"
                "- Machine learning guideline"
            )

        elif matched_title:
            matched_file = next(g for g in uploaded_graphs if g["title"] == matched_title)
            df_path = os.path.join(DATA_DIR, matched_file["filename"])
            df = pd.read_excel(df_path) if df_path.endswith(".xlsx") else pd.read_csv(df_path)

            if "add event" in task_type:
                prompt_message = message.replace(matched_title, "").strip()
                event_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Extract the event type and start/end time in minutes as JSON. "
                                'Example: {"type":"exercise", "start":30, "end":60}'
                            )
                        },
                        {"role": "user", "content": prompt_message}
                    ]
                )
                parsed_event = json.loads(event_response.choices[0].message.content)
                events = json.load(open(EVENT_FILE, encoding="utf-8"))
                events.setdefault(matched_title, []).append(parsed_event)
                json.dump(events, open(EVENT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                reply = f"Event added: {parsed_event}"

            elif "delete event" in task_type:
                prompt_message = message.replace(matched_title, "").strip()

                delete_event_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Extract the event to delete as JSON. "
                                'Example: {"type":"exercise", "start":30, "end":60}'
                            )
                        },
                        {"role": "user", "content": prompt_message}
                    ]
                )

                content = delete_event_response.choices[0].message.content
                try:
                    start_index = content.find("{")
                    end_index = content.rfind("}") + 1
                    json_str = content[start_index:end_index]
                    parsed_event = json.loads(json_str)
                except Exception:
                    parsed_event = {}

                events = json.load(open(EVENT_FILE, encoding="utf-8"))
                event_list = events.get(matched_title, [])

                new_event_list = [
                    ev for ev in event_list
                    if not (
                        ev.get("type") == parsed_event.get("type") and
                        ev.get("start") == parsed_event.get("start") and
                        ev.get("end") == parsed_event.get("end")
                    )
                ]

                events[matched_title] = new_event_list
                json.dump(events, open(EVENT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

                reply = f"Event deleted: {parsed_event}"

            elif "prediction" in task_type:
                result = predict_from_title(matched_title)

                prediction_js = json.dumps({
                    "x": result["x"],
                    "prediction": result["prediction"],
                    "ground_truth": result["ground_truth"]
                })

                reply = f"""
                Prediction result generated.<br>
                <canvas id="chat-prediction-chart" style="margin-top:1rem; max-width: 800px; max-height: 500px;"></canvas>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <script>
                const data = {prediction_js};

                const ctx = document.getElementById("chat-prediction-chart").getContext("2d");
                new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: data.x,
                        datasets: [
                            {{
                                label: 'Prediction',
                                data: data.prediction,
                                borderColor: 'blue',
                                borderWidth: 2,
                                fill: false,
                                tension: 0.3
                            }},
                            {{
                                label: 'Ground Truth',
                                data: data.ground_truth,
                                borderColor: 'red',
                                borderWidth: 2,
                                fill: false,
                                borderDash: [5, 5],
                                tension: 0.3
                            }}
                        ]
                    }},
                    options: {{
                        scales: {{
                            x: {{
                                title: {{
                                    display: true,
                                    text: 'Time (min)'
                                }}
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'TG value'
                                }}
                            }}
                        }}
                    }}
                }});
                </script>
                """

            elif "feature" in task_type or "auc" in message.lower():
                structure_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Extract graph title, start time, end time, and feature names as JSON. "
                                'Example: {"title":"midcarb_1", "start":0, "end":180, "features":["auc"]}'
                            )
                        },
                        {"role": "user", "content": message}
                    ]
                )

                try:
                    content = structure_response.choices[0].message.content
                    print("GPT response:\n", content)

                    start_index = content.find("{")
                    end_index = content.rfind("}") + 1
                    json_str = content[start_index:end_index]
                    parsed = json.loads(json_str)
                    print("Parsed JSON:\n", parsed)

                    feature_names = parsed.get("features", [])
                    start_time = parsed.get("start")
                    end_time = parsed.get("end")
                    feature_title = parsed.get("title", "").lower()

                    matched_file = next((g for g in uploaded_graphs if g["title"].lower() == feature_title), None)
                    if not matched_file:
                        raise Exception(f"No graph found for title '{feature_title}'.")

                    all_features = []
                    for fname in feature_names:
                        df_single = extract_feature(
                            matched_file["biomarker"],
                            matched_file["task_type"],
                            matched_file["disease_status"],
                            fname,
                            start_time,
                            end_time
                        )
                        df_single["subject_id"] = matched_file["subject_id"]
                        all_features.append(df_single.set_index("subject_id"))

                    merged_df = pd.concat(all_features, axis=1).reset_index()
                    feature_set_name = "_".join(feature_names)
                    out_path = os.path.join(DATA_DIR, f"{feature_set_name}.xlsx")
                    merged_df.to_excel(out_path, index=False)

                    label_record = {
                        "name": feature_set_name,
                        "title": matched_file["title"],
                        "biomarker": matched_file["biomarker"],
                        "task_type": matched_file["task_type"],
                        "disease_status": matched_file["disease_status"],
                        "subject_id": matched_file["subject_id"],
                        "start": start_time,
                        "end": end_time
                    }

                    if os.path.exists(FEATURE_LABEL_FILE):
                        labels = json.load(open(FEATURE_LABEL_FILE, encoding="utf-8"))
                    else:
                        labels = []
                    labels.append(label_record)
                    json.dump(labels, open(FEATURE_LABEL_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

                    reply = f"Feature extraction complete: {feature_names} ({start_time} to {end_time} min)"

                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    reply = f"Feature extraction failed: {e}"

            else:
                x = df.iloc[:, 0].astype(float)
                y = df.iloc[:, 1].astype(float)

                auc = np.trapz(y, x)
                slope = np.polyfit(x, y, 1)[0]
                difference = y.max() - y.min()

                head = df.describe().to_markdown()
                extra_stats = (
                    f"- AUC: {auc:.6f}\n"
                    f"- slope: {slope:.6f}\n"
                    f"- difference: {difference:.6f}"
                )

                full_prompt = (
                    f"Statistical summary of '{matched_title}':\n{head}\n\n"
                    f"Additional calculated features:\n{extra_stats}\n\n"
                    f"Request: {message}"
                )
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": "You are a helpful assistant."}] +
                             session_chat_history + [{"role": "user", "content": full_prompt}]
                )
                reply = response.choices[0].message.content.strip()
        else:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a helpful assistant."}] +
                         session_chat_history + [{"role": "user", "content": message}]
            )
            reply = response.choices[0].message.content.strip()

        assistant_entry = {
            "role": "assistant",
            "content": reply,
            "timestamp": timestamp,
            "type": (
                "Guideline" if "guideline" in task_type else
                "Event" if "event" in task_type else
                "Prediction" if "prediction" in task_type else
                "Analysis" if "analysis" in task_type else
                "Other"
            )
        }
        session_chat_history.append(assistant_entry)

        full_log = chat_history if os.path.exists(CHAT_LOG_FILE) else []
        full_log.append(user_entry)
        full_log.append(assistant_entry)
        save_chat_log(full_log)

    except Exception as e:
        reply = f"Error occurred: {str(e)}"
        session_chat_history.append({"role": "assistant", "content": reply, "timestamp": timestamp})

    events = json.load(open(EVENT_FILE, encoding="utf-8"))
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "tab": "chat",
        "response": reply,
        "chat_history": sorted(session_chat_history, key=lambda x: x["timestamp"], reverse=True),
        "uploaded_graphs": uploaded_graphs,
        "matched_title": matched_title,
        "events": events
    })


@app.post("/extract_feature")
async def extract_feature_form(
    request: Request,
    biomarker: str = Form(...),
    task_type: str = Form(...),
    disease_status: str = Form(...),
    start_time: int = Form(...),
    end_time: int = Form(...),
    feature_name: List[str] = Form(...),
    subject_id: str = Form(...),
    title: str = Form(...)
):
    try:
        if isinstance(feature_name, str):
            feature_name = json.loads(feature_name)

        feature_set_name = "_".join(feature_name)
        all_features = []

        for fname in feature_name:
            df_single = extract_feature(biomarker, task_type, disease_status, fname, start_time, end_time)
            df_single["subject_id"] = subject_id
            all_features.append(df_single.set_index("subject_id"))

        merged_df = pd.concat(all_features, axis=1).reset_index()

        out_path = os.path.join(DATA_DIR, f"{feature_set_name}.xlsx")
        merged_df.to_excel(out_path, index=False)

        label_record = {
            "name": feature_set_name,
            "title": title,
            "biomarker": biomarker,
            "task_type": task_type,
            "disease_status": disease_status,
            "start": start_time,
            "end": end_time,
            "subject_id": subject_id
        }

        if os.path.exists(FEATURE_LABEL_FILE):
            feature_labels = json.load(open(FEATURE_LABEL_FILE, encoding="utf-8"))
        else:
            feature_labels = []

        feature_labels.append(label_record)
        json.dump(feature_labels, open(FEATURE_LABEL_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        return RedirectResponse(url="/?tab=featureex", status_code=303)

    except Exception as e:
        return HTMLResponse(content=f"<h1>Feature extraction failed: {e}</h1>", status_code=500)


@app.post("/delete_event")
async def delete_event(request: Request):
    try:
        data = await request.json()
        title = data.get("title")
        event_to_delete = data.get("event")

        if not (title and event_to_delete):
            return JSONResponse(content={"error": "title and event are required."}, status_code=400)

        with open(EVENT_FILE, encoding="utf-8") as f:
            events = json.load(f)

        if title not in events:
            return JSONResponse(content={"error": "No events found for this title."}, status_code=404)

        original_len = len(events[title])
        events[title] = [
            ev for ev in events[title]
            if not (
                ev["type"] == event_to_delete["type"] and
                ev["start"] == event_to_delete["start"] and
                ev["end"] == event_to_delete["end"]
            )
        ]
        deleted = len(events[title]) < original_len

        with open(EVENT_FILE, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)

        return JSONResponse(content={"message": "Delete completed" if deleted else "No matching event found"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/delete_chat_log")
async def delete_chat_log():
    global chat_history, session_chat_history
    chat_history = []
    session_chat_history = []
    save_chat_log([])
    return RedirectResponse(url="/?tab=history", status_code=303)


@app.get("/features")
async def get_features():
    if os.path.exists(FEATURE_LABEL_FILE):
        features = json.load(open(FEATURE_LABEL_FILE, encoding="utf-8"))
    else:
        features = []
    return features


@app.get("/feature_data/{feature_name}")
async def get_feature_data(feature_name: str):
    path = os.path.join(DATA_DIR, f"{feature_name}.xlsx")
    if not os.path.exists(path):
        return {"error": "File does not exist."}
    df = pd.read_excel(path)
    return {
        "columns": df.columns.tolist(),
        "rows": df.fillna("").astype(str).values.tolist()
    }


@app.post("/delete_feature")
async def delete_feature(request: Request):
    data = await request.json()
    feature_name = data.get("feature_name")

    if not feature_name:
        return JSONResponse(content={"error": "Feature name is required"}, status_code=400)

    feature_file_path = os.path.join(DATA_DIR, f"{feature_name}.xlsx")
    feature_json_path = FEATURE_LABEL_FILE

    if os.path.exists(feature_json_path):
        with open(feature_json_path, "r", encoding="utf-8") as f:
            features = json.load(f)
        features = [f for f in features if f["name"] != feature_name]
        with open(feature_json_path, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=2)

    if os.path.exists(feature_file_path):
        os.remove(feature_file_path)

    return JSONResponse(content={"message": "Delete completed"})


@app.post("/train_model")
async def train_model(selected_titles: List[str] = Form(...)):
    try:
        result = train_model_from_titles(selected_titles)
        return JSONResponse(content={
            "status": "success",
            "message": result["message"]
        })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=400)


@app.post("/predict")
async def predict_api(title: str = Form(...)):
    try:
        result = predict_from_title(title)
        return JSONResponse(content={
            "status": "success",
            "title": result["title"],
            "prediction": result["prediction"],
            "ground_truth": result["ground_truth"],
            "x": result["x"]
        })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=400)


@app.post("/chat_api")
async def chat_api(
    message: str = Form(...),
    image: UploadFile | None = File(None),
):
    global chat_history

    if re.search(r"(glucose scheduler|schedule|plan|scheduler)", message, re.IGNORECASE):
        return JSONResponse({
            "status": "redirect",
            "redirect_url": "/?tab=scheduler"
        })

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    matched_title = extract_title_from_message(message, uploaded_graphs)

    image_url = None
    local_image_path = None

    try:
        if image and image.filename:
            if not (image.content_type or "").startswith("image/"):
                raise ValueError("Only image files can be uploaded.")

            content = await image.read()
            if len(content) > 10 * 1024 * 1024:
                raise ValueError("Image size exceeds 10MB.")

            ext = os.path.splitext(image.filename)[1].lower()
            if ext not in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"]:
                ext = ".png"

            fname = f"{uuid.uuid4().hex}{ext}"
            fpath = os.path.join(UPLOAD_DIR, fname)

            with open(fpath, "wb") as f:
                f.write(content)

            if not is_valid_image(fpath):
                os.remove(fpath)
                raise ValueError("Invalid image file.")

            local_image_path = fpath
            image_url = f"/{fpath.replace(os.sep, '/')}"

        user_entry = {
            "role": "user",
            "content": message,
            "timestamp": timestamp,
            "image_url": image_url
        }
        session_chat_history.append(user_entry)

        sys_prompt = (
            "You are a helpful assistant for a health monitoring dashboard. "
            "Be concise, factual, and practical. "
            "If the user uploaded an image, analyze what is visible in the image and answer accordingly. "
            "If the user mentions a chart title, use the provided context summary to ground your answer."
        )

        context_msgs = []
        if matched_title:
            try:
                matched_file = next(g for g in uploaded_graphs if g["title"] == matched_title)
                df_path = os.path.join(DATA_DIR, matched_file["filename"])
                df = pd.read_excel(df_path) if df_path.endswith(".xlsx") else pd.read_csv(df_path)

                x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
                y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
                valid = x.notna() & y.notna()
                x, y = x[valid], y[valid]

                stats = df.describe().to_string()

                try:
                    auc = float(np.trapz(y.values, x.values))
                except Exception:
                    auc = None
                try:
                    slope = float(np.polyfit(x.values, y.values, 1)[0])
                except Exception:
                    slope = None
                try:
                    diff = float(y.max() - y.min())
                except Exception:
                    diff = None

                ctx_text = (
                    f"[DATA SUMMARY for '{matched_title}']\n"
                    f"- biomarker: {matched_file['biomarker']}, task: {matched_file['task_type']}, disease: {matched_file['disease_status']}\n"
                    f"- table describe():\n{stats}\n"
                    f"- derived: AUC={auc}, slope={slope}, difference={diff}\n"
                )
                context_msgs.append({"role": "system", "content": ctx_text})
            except Exception:
                pass

        history_tail = session_chat_history[-20:]
        msgs = [{"role": "system", "content": sys_prompt}]
        msgs += context_msgs

        for h in history_tail:
            if h.get("content"):
                role = "user" if h["role"] == "user" else "assistant"
                msgs.append({"role": role, "content": h["content"]})

        if image_url:
            full_image_url = f"http://127.0.0.1:8000{image_url}"
            msgs.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {
                        "type": "image_url",
                        "image_url": {"url": full_image_url}
                    }
                ]
            })
        else:
            msgs.append({"role": "user", "content": message})

        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0.7,
        )

        reply = (result.choices[0].message.content or "").strip()

        assistant_entry = {
            "role": "assistant",
            "content": reply,
            "timestamp": timestamp,
            "type": "Other"
        }
        session_chat_history.append(assistant_entry)

        full_log = chat_history if os.path.exists(CHAT_LOG_FILE) else []
        full_log.append(user_entry)
        full_log.append(assistant_entry)
        save_chat_log(full_log)

        return JSONResponse({
            "status": "ok",
            "assistant": assistant_entry,
            "user": user_entry
        })

    except Exception as e:
        err = {
            "role": "assistant",
            "content": f"Error: {e}",
            "timestamp": timestamp,
            "type": "Other"
        }
        session_chat_history.append(err)
        return JSONResponse({"status": "error", "assistant": err}, status_code=500)


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})