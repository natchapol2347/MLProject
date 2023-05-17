const webcam = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const captureButton = document.getElementById("capture");
const resultParagraph = document.getElementById("result");
const resultPoem = document.getElementById("results");
const cropButton = document.getElementById("crop");
const newImage = document.getElementById('capimage');
const applyButton = document.getElementById('applybutton')
const generateButton = document.getElementById('generate')

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
  // var link = document.createElement("a");
  // link.download = "image.png";

  canvas.toBlob((blob) => {
    var reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.onloadend = function () {
      var base64String = reader.result;
      newImage.src = base64String;
    }
  }
  );
});



generateButton.addEventListener("click", async () => {
  // Get ArrayBuffer Image
  const response = await fetch(newImage.src);
  const arrayBuffer = await response.arrayBuffer();
  const byteArray = new Uint8Array(arrayBuffer);
  console.log('Byte Array:', byteArray);

  // Convert byte array to base64-encoded string
  const base64String = await convertByteArrayToBase64(byteArray);

  const payload = { image: base64String };
  const header = { 'Content-Type': 'multipart/form-data' };
  const config = { headers: header };
  console.log('formData:', payload);

  try {
    const response = await axios.post("http://127.0.0.1:5000/api/recognize", payload, config);
    console.log(response.data);
    resultParagraph.textContent = `Predicted word: ${response.data.word}`;
    resultPoem.textContent = `Poem: ${response.data.poem}`;
  } catch (err) {
    console.error("Error calling Flask API: ", err);
    resultParagraph.textContent = "Error: Failed to recognize the word";
  }
});

function convertByteArrayToBase64(byteArray) {
  return new Promise((resolve, reject) => {
    const fileReader = new FileReader();
    fileReader.onloadend = function () {
      const base64String = fileReader.result.split(",")[1];
      resolve(base64String);
    };
    fileReader.onerror = function (error) {
      reject(error);
    };
    fileReader.readAsDataURL(new Blob([byteArray]));
  });
}





cropButton.addEventListener("click", () => {

  const cropper = new Cropper(newImage, {
    aspectRatio: 0,
    viewMode: 0,
  });

  applyButton.addEventListener("click", () => {
    var crop = cropper.getCroppedCanvas().toDataURL("image/png");
    newImage.src = crop;
    cropper.destroy();
  });


  // link.click()



});