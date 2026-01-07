document
  .getElementById("startCamera")
  .addEventListener("click", async function () {
    const cameraContainer = document.getElementById("cameraContainer");
    const videoFeed = document.getElementById("videoFeed");
    const selectedModel = document.getElementById("model_choice").value;
    const currentModelDiv = document.getElementById("currentModel");

    const getModelDisplayName = (modelValue) => {
      const modelMap = {
        naive_bayes: "Naive Bayes",
        svm: "Support Vector Machine",
        random_forest: "Random Forest",
        logistic_regression: "Logistic Regression",
        knn: "K-Nearest Neighbors",
      };
      return modelMap[modelValue] || modelValue;
    };

    if (cameraContainer.style.display === "none") {
      try {
        await fetch("/stop_camera").then((res) => res.json());

        videoFeed.src = "";

        const formData = new FormData();
        formData.append("model_choice", selectedModel);

        const response = await fetch("/start_camera", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();

        if (data.status === "success") {
          videoFeed.src = `/video_feed?t=${new Date().getTime()}`;
          cameraContainer.style.display = "block";
          this.textContent = "Stop Camera";
          currentModelDiv.textContent = `Model: ${getModelDisplayName(
            selectedModel
          )}`;
        }
      } catch (error) {
        console.error("Error:", error);
      }
    } else {
      try {
        const response = await fetch("/stop_camera");
        const data = await response.json();

        if (data.status === "success") {
          cameraContainer.style.display = "none";
          videoFeed.src = ""; 
          this.textContent = "Start Camera Detection";
          currentModelDiv.textContent = "";
        }
      } catch (error) {
        console.error("Error:", error);
      }
    }
  });

document
  .getElementById("model_choice")
  .addEventListener("change", async function () {
    const cameraContainer = document.getElementById("cameraContainer");
    const videoFeed = document.getElementById("videoFeed");

    if (cameraContainer.style.display !== "none") {
      await fetch("/stop_camera").then((res) => res.json());
      cameraContainer.style.display = "none";
      videoFeed.src = ""; 
      document.getElementById("startCamera").textContent =
        "Start Camera Detection";
    }
  });
