import React, { useState } from "react";
import "./SkinHealthDashboard.css";

export default function App() {

  const [uploadedImage, setUploadedImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);

  const [formData, setFormData] = useState({
    sleep: "",
    stress: "Low",
    water: "",
    exercise: "",
    screenTime: ""
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [riskLevel, setRiskLevel] = useState(null);
const [recommendation, setRecommendation] = useState("");
  const handleImageUpload = (e) => {
    const file = e.target.files[0];

    if (file) {
      setImageFile(file);

      const reader = new FileReader();
      reader.onload = (event) => {
        setUploadedImage(event.target.result);
      };

      reader.readAsDataURL(file);
    }
  };

  const handleFormChange = (e) => {
    const { name, value } = e.target;

    setFormData((prev) => ({
      ...prev,
      [name]: value
    }));
  };

  const handleAnalyze = async () => {

  if (!imageFile) {
    alert("Please upload an image");
    return;
  }

  if (
    !formData.sleep ||
    !formData.water ||
    !formData.exercise ||
    !formData.screenTime
  ) {
    alert("Please fill all lifestyle fields");
    return;
  }

  setLoading(true);

  const data = new FormData();

  data.append("file", imageFile);
  data.append("sleep", formData.sleep);
  data.append("stress", formData.stress);
  data.append("water", formData.water);
  data.append("exercise", formData.exercise);
  data.append("screen_time", formData.screenTime);

  try {

    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: data
    });

    const result = await response.json();

    setPrediction(result.skin_condition);
    setRiskLevel(result.risk_level);
    setRecommendation(result.recommendation);

  } catch (error) {

    console.error(error);
    alert("Prediction failed");

  }

  setLoading(false);
};

  return (
    <div className="dashboard-container">

      {/* Header */}

      <header className="dashboard-header">
        <div className="header-content">
          <div className="header-icon">✨</div>
          <div>
            <h1>AI Skin Health Analyzer</h1>
            <p>Advanced AI-powered skin analysis with lifestyle insights</p>
          </div>
        </div>
      </header>


      {/* MAIN GRID */}

      <main className="dashboard-grid">

        {/* Upload Card */}

        <div className="card upload-card">

          <div className="card-header">
            <h2>📸 Upload Skin Image</h2>
            <div className="accent-line"></div>
          </div>

          <div className="upload-area">

            <label htmlFor="image-input" className="upload-input-label">

              <div className="upload-icon">📤</div>

              <p className="upload-text">
                Click to upload or drag image
              </p>

              <p className="upload-subtext">
                PNG, JPG up to 10MB
              </p>

            </label>

            <input
              id="image-input"
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="upload-input"
            />

          </div>


          {uploadedImage && (

            <div className="image-preview-container">

              <img
                src={uploadedImage}
                alt="Uploaded skin"
                className="image-preview"
              />

              <p className="preview-status">✓ Image uploaded</p>

            </div>

          )}


          <button
            className="analyze-btn"
            onClick={handleAnalyze}
            disabled={!uploadedImage}
          >
            {loading ? "Analyzing..." : "🚀 Analyze Skin"}
          </button>

        </div>



        {/* Lifestyle Questionnaire */}

        <div className="card questionnaire-card">

          <div className="card-header">
            <h2>💚 Lifestyle Profile</h2>
            <div className="accent-line"></div>
          </div>

          <form className="form-grid">

            <div className="form-group">
              <label>Sleep Hours / Night</label>
              <input
                type="number"
                name="sleep"
                value={formData.sleep}
                onChange={handleFormChange}
                className="input-field"
              />
            </div>


            <div className="form-group">
              <label>Stress Level</label>
              <select
                name="stress"
                value={formData.stress}
                onChange={handleFormChange}
                className="input-field"
              >
                <option>Low</option>
                <option>Medium</option>
                <option>High</option>
              </select>
            </div>


            <div className="form-group">
              <label>Water Intake (L/day)</label>
              <input
                type="number"
                name="water"
                value={formData.water}
                onChange={handleFormChange}
                className="input-field"
              />
            </div>


            <div className="form-group">
              <label>Exercise (min/week)</label>
              <input
                type="number"
                name="exercise"
                value={formData.exercise}
                onChange={handleFormChange}
                className="input-field"
              />
            </div>


            <div className="form-group">
              <label>Screen Time (hours/day)</label>
              <input
                type="number"
                name="screenTime"
                value={formData.screenTime}
                onChange={handleFormChange}
                className="input-field"
              />
            </div>

          </form>

        </div>



        {/* RESULT CARD */}

{prediction && (

  <div className="card result-card">

    <div className="card-header">
      <h2>🔬 Analysis Results</h2>
      <div className="accent-line"></div>
    </div>

    <div className="results-grid">

      <div className="result-item">
        <div className="result-label">Skin Condition</div>

        <div className="result-value healthy">
          {prediction}
        </div>

        <p className="result-description">
          AI detected skin condition
        </p>
      </div>


      <div className="result-item">
        <div className="result-label">Risk Level</div>

        <div className="result-value score">
          {riskLevel}
        </div>

        <p className="result-description">
          Lifestyle risk prediction
        </p>
      </div>

    </div>

    <div className="ai-advice">

      <h3>💡 AI Recommendation</h3>

      <p>{recommendation}</p>

    </div>

  </div>

)}
      </main>

    </div>
  );
}