import json
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from .model import get_classifier

app = FastAPI()


# Serve UI from static file
with open("static/index.html", "r", encoding="utf-8") as f:
    INDEX_HTML = f.read()


@app.get("/")
async def get_root():
    return HTMLResponse(INDEX_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = []

    try:
        while True:
            data = await websocket.receive_text()
            parsed_data = json.loads(data)
            audio_chunk = np.array(parsed_data['audio'], dtype=np.float32)
            input_timestamp = parsed_data.get('timestamp')
            buffer.extend(audio_chunk)

            if len(buffer) >= 48000:
                audio = np.array(buffer[-48000:])
                audio_level = np.abs(audio).mean()

                if audio_level > 0.01:
                    audio = audio / max(np.abs(audio).max(), 1e-5)

                    classifier = get_classifier()
                    result = classifier(audio, sampling_rate=16000)

                    sorted_emotions = sorted(result, key=lambda x: x['score'], reverse=True)

                    await websocket.send_json({
                        "emotion": sorted_emotions[0]['label'],
                        "confidence": sorted_emotions[0]['score'],
                        "allEmotions": sorted_emotions,
                        "silence": False,
                        "level": int(min(audio_level * 1000, 100)),
                        "inputTimestamp": input_timestamp
                    })
                else:
                    await websocket.send_json({
                        "silence": True,
                        "level": int(min(audio_level * 1000, 100))
                    })
    except Exception as e:
        print(f"Error: {e}")
