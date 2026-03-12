from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# CNN prediction function
from main import predict

# ML model prediction
from models.lifestyle_predict import predict_risk

# Recommendation engine
from models.recommendation import get_recommendation


app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Skin Health API Running"}


@app.post("/predict")
async def predict_skin(
    file: UploadFile = File(...),

    # Lifestyle inputs from UI
    sleep: float = Form(...),
    stress: str = Form(...),
    water: float = Form(...),
    exercise: float = Form(...),
    screen_time: float = Form(...)
):

    # Read image
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    image_path = "temp.jpg"
    image.save(image_path)

    # CNN prediction
    skin_condition = predict(image_path)

    # ML prediction
    risk_level = predict_risk(
        sleep,
        stress,
        water,
        exercise,
        screen_time,
        skin_condition
    )

    # Recommendation
    recommendation = get_recommendation(risk_level)

    return {
        "skin_condition": skin_condition,
        "risk_level": risk_level,
        "recommendation": recommendation
    }