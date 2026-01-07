document.addEventListener("DOMContentLoaded", function () {
  const modelChoiceElement = document.getElementById("model_choice");

  for (const model in modelNames) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model; 
    modelChoiceElement.appendChild(option);
  }

  modelChoiceElement.value = "naive_bayes"; 

  localStorage.removeItem("selectedModel");
});

document
  .getElementById("model_choice")
  .addEventListener("change", async function () {
    const selectedModel = this.value;

    try {
      const response = await fetch("/start_camera", {
        method: "POST",
        body: new URLSearchParams({ model_choice: selectedModel }),
      });

      const data = await response.json();
      if (data.status === "success") {
        console.log("Model đã được thay đổi thành công!");
      } else {
        console.error("Lỗi:", data.error);
      }
    } catch (error) {
      console.error("Error:", error);
    }
  });

document
  .getElementById("startCamera")
  .addEventListener("click", async function () {
    const selectedModel = document.getElementById("model_choice").value;

    const formData = new FormData();
    formData.append("model_choice", selectedModel);

    const response = await fetch("/start_camera", {
      method: "POST",
      body: formData,
    });

    const currentModelDiv = document.getElementById("currentModel");
    currentModelDiv.textContent = `Model: ${modelNames[selectedModel]}`; 
  });
