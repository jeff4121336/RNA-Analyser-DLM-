const modelDescriptions = [
  {
    id: "m1",
    title: "13-Class Single Label Model",
    description:
      "This model predicts the class of an input RNA sequence. The possible RNA types \
		are 5S rRNA, 5.8S rRNA, tRNA, ribozyme, CD-box, miRNA, Intron gpI, Intron gpII, scaRNA, HACA-box, \
		riboswitch, IRES, and leader, totaling 13 ncRNA classes. The prediction includes the class label, \
		prediction score, and feature maps.",
  },
  {
    id: "m2",
    title: "14-Class Single Label Model",
    description:
      "This model predicts the class of an input RNA sequence. The possible RNA types \
		are 5S rRNA, 5.8S rRNA, tRNA, ribozyme, CD-box, miRNA, Intron gpI, Intron gpII, scaRNA, HACA-box, \
		riboswitch, IRES, leader, and mRNA, making a total of 14 classes. The prediction includes the class label, \
		prediction score, and feature maps.",
  },
];

function generateRandomId(prefix) {
  return `${prefix}-${Math.random().toString(36).substring(0, 16)}`; // Generates a random string
}

function handleModelClick(event) {
  const clickedModelId = event.currentTarget.id; // Get the ID of the clicked div
  const modelInfo = document.querySelector(".model-info"); // Get the model-info div
  const selectedModel = modelDescriptions.find(
    (model) => model.id === clickedModelId
  );
  console.log(selectedModel);
  console.log(modelInfo);
  console.log(clickedModelId);

  if (!selectedModel) {
    modelInfo.innerHTML = `<p>No description available for this model.</p>`;
    return;
  }

  modelInfo.innerHTML = `
		<div style="margin: 10px;">
		<p style="font-size: 16px; font-weight: bold;">Selected: ${selectedModel.title}</p>
			<p>${selectedModel.description}</p>
		</div>
	`;

  fetch("/change", {
    method: "POST",
    headers: {
      "Content-Type": "application/json", // Specify that you're sending JSON
    },
    body: JSON.stringify({ model: clickedModelId }), // Convert the data to a JSON string
  })
    .then((response) => response.json()) // Parse the JSON response
    .then((data) => {
      console.log("Success:", data); // Log the updated modelSelected value
    })
    .catch((error) => console.error("Error:", error));
}

function displayHeatmap(id) {
  const heatmapContainer = document.getElementById(id);

  const existingImages = heatmapContainer.getElementsByTagName("img");
  if (existingImages.length > 0) {
    // If images exist, change their visibility
    Array.from(existingImages).forEach((img) => {
      img.style.visibility = "visible";
    });
    return;
  }

  fetch("/api/images")
    .then((response) => response.json())
    .then((images) => {
      console.log(id);
      images.forEach((imageFilename) => {
        const heatmapImage = document.createElement("img");
        heatmapImage.src = `static/images/${imageFilename}`;
        heatmapImage.alt = "Heatmap";
        heatmapContainer.appendChild(heatmapImage);
      });
    });
}

function hideHeatmap(id) {
  const heatmapContainer = document.getElementById(id);
  const heatmapImages = heatmapContainer.getElementsByClassName("img");
  Array.from(heatmapImages).forEach((img) => {
    img.style.visibility = "hidden"; // Hide the image
  });
}

const models = document.querySelectorAll(".model-selection-clickable");
models.forEach((model) => {
  model.addEventListener("click", handleModelClick);
});

document
  .getElementById("prompt-send")
  .addEventListener("click", function (event) {
    event.preventDefault(); // Prevent form submission
    const userInput = document.getElementById("user-input").value;
    const container = document.getElementById("chatroom");
    var lastMessage = container.querySelector(".row.message:last-of-type");

    const newUserMessage = document.createElement("div");
    newUserMessage.className = "row message user-message";
    newUserMessage.textContent = userInput;

    container.insertBefore(newUserMessage, lastMessage.nextSibling);
    // Send the input to the /predict route using fetch
    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input: userInput }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        // Create the prediction message\
        var display = false;

        document.getElementById("user-input").value = "";

        const predictionMessage = `Prediction:</br> ${data.prediction} </br> Min-Max Scaling Prediction Score: </br> ${data.score}`;

        const newPredictionMessage = document.createElement("div");
        newPredictionMessage.className = "row message bot-message";
        newPredictionMessage.innerHTML = predictionMessage;

        const showMoreDiv = document.createElement("div");
        showMoreDiv.id = generateRandomId("heatmap");
        showMoreDiv.className = "row message bot-message";
        showMoreDiv.style.color = "grey";
        showMoreDiv.style.cursor = "pointer";
        showMoreDiv.style.fontSize = "14px";
        showMoreDiv.innerHTML = `&nbsp; show more ...`;
        console.log(showMoreDiv.id);

        showMoreDiv.addEventListener("mouseenter", function () {
          showMoreDiv.style.color = "darkblue";
        });
        showMoreDiv.addEventListener("mouseleave", function () {
          showMoreDiv.style.color = "grey";
        });
        showMoreDiv.addEventListener("click", function () {
          if (display == false) {
            display = true;
            displayHeatmap(showMoreDiv.id);
            showMoreDiv.innerHTML = `&nbsp collapse ...`;
          } else {
            display = false;
            showMoreDiv.innerHTML = `&nbsp show more ...`;
            hideHeatmap(showMoreDiv.id);
          }
        });

        // Find the container and append the new message after the last message
        lastMessage = container.querySelector(".row.message:last-of-type");
        container.insertBefore(newPredictionMessage, lastMessage.nextSibling); // Insert after the last message
        lastMessage = container.querySelector(".row.message:last-of-type");
        container.insertBefore(showMoreDiv, lastMessage.nextSibling); // Insert after the last message
      })
      .catch((error) => {
        console.error("There was a problem with the fetch operation:", error);
      });
  });
