const webcam = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const captureButton = document.getElementById("capture");
const resultParagraph = document.getElementById("result");

// Access the user's webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        webcam.srcObject = stream;
        webcam.addEventListener("loadedmetadata", () => {
            webcam.play();
        });
    })
    .catch(err => {
        console.error("Error accessing webcam: ", err);
    });

// Capture a frame from the webcam and send it to the Flask API
captureButton.addEventListener("click", () => {
    canvas.getContext("2d").drawImage(webcam, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(async blob => {
        const formData = new FormData();
        formData.append("image", blob);

        try {
            const response = await axios.post("http://127.0.0.1:5000/api/recognize", formData);
            console.log(response.data)
            resultParagraph.textContent = `Predicted word: ${response.data.word}`;
        } catch (err) {
            console.error("Error calling Flask API: ", err);
            resultParagraph.textContent = "Error: Failed to recognize the word";
        }
    });
});
