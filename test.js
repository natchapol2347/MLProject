const axios = require('axios');

async function test() {
  try {
    // Send the FormData object to the Flask app using axios
    var response = await axios.post('http://127.0.0.1:5000/api/recognize');

    // Log the response data
    console.log(response.data);
  } catch (err) {
    console.error('Error calling Flask API:', err);
  }
  return response.data
}
test()