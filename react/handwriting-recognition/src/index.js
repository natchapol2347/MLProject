import React, { useState, useEffect,useRef } from 'react';
import ReactDOM from 'react-dom';
import axios from 'axios';
import './index.css';
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
const App = () => {
  const [prediction, setPrediction] = useState('');
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  
  useEffect(() => {
    // Access the user's camera and render the stream in a video element
    const video = videoRef.current;
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        video.srcObject = stream;
        video.play();
      })
      .catch((err) => {
        console.log('Error accessing camera: ', err);
      });
  }, []);

  const captureImage = async () => {
    console.log('eeje')
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const image = canvas.toDataURL();
    const formData = new FormData();
    formData.append('image', image);
    try {
      const response = await axios.post('http://localhost:5000/api/recognize', formData);
      setPrediction(response.data.word);
    } catch (error) {
      console.error('Error calling Flask API: ', error);
      setPrediction('Error: Failed to recognize the word');
    }
  }

  return (
    <div className="container">
      <h1>Handwriting Recognition</h1>
      <video id="webcam" ref={videoRef} autoPlay></video>
      <canvas id="canvas" ref={canvasRef} width="640" height="480"></canvas>
      <br />
      <button onClick={() => captureImage()}>Capture</button>
      <p id="result">Predicted word: {prediction}</p>
    </div>
  );
}
