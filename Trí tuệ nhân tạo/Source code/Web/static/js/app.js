document.getElementById("predictButton").addEventListener("click", function () {
  const formData = new FormData(document.getElementById("uploadForm"));

  fetch("/predict", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      const resultElement = document.getElementById("result");
      if (data.error) {
        resultElement.innerText = `Error: ${data.error}`;
      } else if (data.emotion) {
        const sortedProbabilities = Object.entries(data.probabilities)
          .filter(([emotion, prob]) => prob > 0)
          .sort((a, b) => b[1] - a[1]);

        const resultText =
          sortedProbabilities.length > 0
            ? sortedProbabilities
                .map(
                  ([emotion, prob]) => `${emotion}: ${(prob * 100).toFixed(2)}%`
                )
                .join("\n")
            : "No emotion detected.";

        resultElement.innerText = `Predicted Emotion:\n${resultText}`;
        resultElement.style.textAlign = "center";
      } else {
        resultElement.innerText = "No emotion detected.";
      }
    })
    .catch((error) => {
      console.error("Error:", error);
    });
});

document.getElementById("media").addEventListener("change", function () {
  const label = document.querySelector(".label-file");
  if (this.files.length > 0) {
    label.classList.remove("file-button");
  } else {
    label.classList.add("file-button");
  }
});
