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
    var link = document.createElement("a");
    link.download = "image.png";

    // canvas.toBlob(function(blob){
    //   link.href = URL.createObjectURL(blob);
    //   console.log(blob);
    //   console.log(link.href); // this line should be here
    //   window.open(link.href)
    // },'image/png');
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
    // canvas.toBlob(async blob => {

    //     // console.log('blob generated:', blob);
        
    //     const formData = {}
    //     link.href = URL.createObjectURL(blob);
    //     console.log(blob);
    //     console.log(link.href); // this line should be here
    //     // window.open(link.href)
    //     let byte = await convertblobtobytes(blob);
    //     console.log(typeof(byte));
    //     const decoder = new TextDecoder();
    //     const str = decoder.decode(byte);
    //     // console.log(str);
    //     // link.click()
    //     const byteString = btoa(String.fromCharCode.apply(null, byte));
    //     byte = `${byte}`
    //     console.log(byteString)
    //     formData.image = byteString;
    //     const header = {
    //         'Content-Type': 'multipart/form-data'
    //     }
    //       const config = { headers: header }
    //       console.log('formData:', formData);
    //     try {
    //         const response = await axios.post("http://127.0.0.1:5000/api/recognize", formData,config);
    //         console.log(response.data)
    //         resultParagraph.textContent = `Predicted word: ${response.data.word}`;
    //     } catch (err) {
    //         console.error("Error calling Flask API: ", err);
    //         resultParagraph.textContent = "Error: Failed to recognize the word";
    //     }
    // });

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
      } catch (err) {
          console.error("Error calling Flask API: ", err);
          resultParagraph.textContent = "Error: Failed to recognize the word";
      }
      };
    }, 'image/jpeg', 0.95);
    // link.click();
    
});
