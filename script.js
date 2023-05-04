const webcam = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const captureButton = document.getElementById("capture");
const resultParagraph = document.getElementById("result");
const resultPoem = document.getElementById("results");

// Access the user's webcam
navigator.mediaDevices.getUserMedia({ 
  video: { 
    width: 850, 
    height: 480 
} 
 })
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
    var link = document.createElement("a");
    link.download = "image.png";

    function convertblobtobytes(blob) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const bytes = new Uint8Array(reader.result);
          resolve(bytes);
        };
        reader.onerror = reject;
        reader.readAsArrayBuffer(blob);
      });
    }
    canvas.toBlob(async blob => {
      const reader = new FileReader();
      reader.readAsArrayBuffer(blob);
      reader.onload = async () => {
        const byteArray = new Uint8Array(reader.result);
        const byteString = btoa(String.fromCharCode.apply(null, byteArray));
        const payload = { image: byteString };
        const header = {'Content-Type': 'multipart/form-data'}
        const config = { headers: header }
        console.log('formData:', payload);
      try {
          const response = await axios.post("http://127.0.0.1:5000/api/recognize", payload,config);
          console.log(response.data)
          resultParagraph.textContent = `Predicted word: ${response.data.word}`;
          resultPoem.textContent = `Poem: ${response.data.poem}`;
      } catch (err) {
          console.error("Error calling Flask API: ", err);
          resultParagraph.textContent = "Error: Failed to recognize the word";
      }
      };
    }, 'image/jpeg', 0.95);
    // link.click();
    
});
