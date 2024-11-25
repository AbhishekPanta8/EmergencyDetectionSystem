const validateBtn = document.getElementById('validate-btn');
const textInput = document.getElementById('text-input');
const loader = document.getElementById('symbol-loader');
const result = document.getElementById('result');

validateBtn.addEventListener('click', async () => {
  const text = textInput.value.trim();
  if (!text) {
    alert("Please enter emergency details.");
    return;
  }

  loader.style.display = 'flex';
  result.innerHTML = '';

  try {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        tweets: [text]
      })
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();

    // Mapping backend results to predefined emoji sets
    const emergencyLevels = {
      'HIGH_EMERGENCY': { level: "High Emergency", emoji: "🔴" },
      'MODERATE_EMERGENCY': { level: "Moderate Emergency", emoji: "🟠" },
      'LOW_EMERGENCY': { level: "Low Emergency", emoji: "🟢" },
      'NON_EMERGENCY': { level: "Non Emergency", emoji: "✅" }
    };

    const disasterTypes = {
      'EARTHQUAKE': { type: "Earthquake", emoji: "🌍" },
      'FIRE': { type: "Fire", emoji: "🔥" },
      'FLOOD': { type: "Flood", emoji: "🌊" },
      'HURRICANE': { type: "Hurricane", emoji: "🌪️" },
      'TORNADO': { type: "Tornado", emoji: "🌪️" },
      'TSUNAMI': { type: "Tsunami", emoji: "🌊" },
      'STORM': { type: "Storm", emoji: "🌩️" },
      'LANDSLIDE': { type: "Landslide", emoji: "🏔️" },
      'VOLCANIC_ERUPTION': { type: "Volcanic Eruption", emoji: "🌋" },
      'OTHER': { type: "Other", emoji: "❓" }
    };

    const sentiments = {
      'URGENT': { sentiment: "Urgent", emoji: "⚡" },
      'PANIC': { sentiment: "Panic", emoji: "😱" },
      'FEAR': { sentiment: "Fear", emoji: "😨" },
      'NEUTRAL': { sentiment: "Neutral", emoji: "😐" }
    };

    const prediction = data.predictions[0].predictions;

    // Display results with actual backend data
    result.innerHTML = `
      <div>🚦 Analysis Complete! 🚦</div>
      <div class="output-columns">
        <div class="column">
          <h3>Emergency Level</h3>
          <p class="output-item">${emergencyLevels[prediction.emergency_level].emoji} ${emergencyLevels[prediction.emergency_level].level}</p>
        </div>
        <div class="column">
          <h3>Disaster Type</h3>
          <p class="output-item">${disasterTypes[prediction.disaster_type].emoji} ${disasterTypes[prediction.disaster_type].type}</p>
        </div>
        <div class="column">
          <h3>Sentiment</h3>
          <p class="output-item">${sentiments[prediction.sentiment].emoji} ${sentiments[prediction.sentiment].sentiment}</p>
        </div>
      </div>
    `;
  } catch (error) {
    console.error('Error:', error);
    result.innerHTML = `<div>Error processing request: ${error.message}</div>`;
  } finally {
    loader.style.display = 'none';
  }
});