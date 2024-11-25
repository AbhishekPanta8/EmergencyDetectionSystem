const validateBtn = document.getElementById('validate-btn');
const textInput = document.getElementById('text-input');
const loader = document.getElementById('symbol-loader');
const result = document.getElementById('result');

validateBtn.addEventListener('click', () => {
  const text = textInput.value.trim();
  if (!text) {
    alert("Please enter emergency details.");
    return;
  }

  loader.style.display = 'flex';
  result.innerHTML = '';

  setTimeout(() => {
    loader.style.display = 'none';

    // Emergency levels
    const emergencyLevels = [
      { level: "High Emergency", emoji: "🔴" },
      { level: "Moderate Emergency", emoji: "🟠" },
      { level: "Low Emergency", emoji: "🟢" },
      { level: "Non Emergency", emoji: "✅" },
      { level: "Potential Emergency", emoji: "⚠️" }
    ];

    // Disaster types
    const disasterTypes = [
      { type: "Earthquake", emoji: "🌍" },
      { type: "Fire", emoji: "🔥" },
      { type: "Flood", emoji: "🌊" },
      { type: "Hurricane", emoji: "🌪️" },
      { type: "Tornado", emoji: "🌪️" },
      { type: "Tsunami", emoji: "🌊" },
      { type: "Other", emoji: "❓" },
      { type: "Storm", emoji: "🌩️" },
      { type: "Landslide", emoji: "🏔️" },
      { type: "Volcanic Eruption", emoji: "🌋" }
    ];

    // Sentiments
    const sentiments = [
      { sentiment: "Panic", emoji: "😱" },
      { sentiment: "Fear", emoji: "😨" },
      { sentiment: "Urgent", emoji: "⚡" },
      { sentiment: "Neutral", emoji: "😐" }
    ];

    // Random selections
    const randomEmergency = emergencyLevels[Math.floor(Math.random() * emergencyLevels.length)];
    const randomDisaster = disasterTypes[Math.floor(Math.random() * disasterTypes.length)];
    const randomSentiment = sentiments[Math.floor(Math.random() * sentiments.length)];

    // Display results with Analysis Complete message
    result.innerHTML = `
      <div>🚦 Analysis Complete! 🚦</div>
      <div class="output-columns">
        <div class="column">
          <h3>Emergency Level</h3>
          <p class="output-item">${randomEmergency.emoji} ${randomEmergency.level}</p>
        </div>
        <div class="column">
          <h3>Disaster Type</h3>
          <p class="output-item">${randomDisaster.emoji} ${randomDisaster.type}</p>
        </div>
        <div class="column">
          <h3>Sentiment</h3>
          <p class="output-item">${randomSentiment.emoji} ${randomSentiment.sentiment}</p>
        </div>
      </div>
    `;
  }, 3000); // Simulate a 3-second delay for loading
});
