// https://github.com/bhattbhavesh91/tf-js-example-1
// https://machinelearningknowledge.ai/learn-image-classification-with-tensorflow-js-using-mobilenet-model-web-app
// https://towardsdatascience.com/deploying-an-image-classifier-using-javascript-84da1480b3a4


CLASSES = {0: "Ampulla of vater",
           1: "Angiectasia",
           2: "Blood - fresh",
           3: "Blood - hematin",
           4: "Erosion",
           5: "Erythema",
           6: "Foreign body",
           7: "Ileocecal valve",
           8: "Lymphangiectasia",
           9: "Normal clean mucosa",
           10: "Polyp",
           11: "Pylorus",
           12: "Reduced mucosal view",
           13: "Ulcer"}

function PreviewImage() {
    var oFReader = new FileReader();
    oFReader.readAsDataURL(document.getElementById("image-selector").files[0]);
    oFReader.onload = function (oFREvent) {
        document.getElementById("selected-image").src = oFREvent.target.result;
        $("#prediction-list").empty();
    };
};

async function getPred(){
	// Read the image and perform preprocessing
    // const img = new Image()
    // img.src='https://image.shutterstock.com/image-vector/vector-illustration-unused-match-stick-260nw-1662505090.jpg';
    // img.crossOrigin = "anonymous";
    let image = $("#selected-image").get(0);
	let tensorImg = tf.browser.fromPixels(image).resizeNearestNeighbor([32, 32]).toFloat().div(tf.scalar(255)).expandDims();
    const predictions = await model.predict(tensorImg).data();
	
	// Get Top5
	let top5 = Array.from(predictions)
    .map(function (p, i) {
        return {
            probability: p,
            className: CLASSES[i]
        };
    }).sort(function (a, b) {
        return b.probability - a.probability;
    }).slice(0, 5);
	
	// Display the results
    top5.forEach(function (p) {
        const probability = (p.probability * 100).toFixed(2);
        $("#prediction-list").append("<li>" + p.className + " : " + probability + " %</li>")
    });
}

async function loadModel() {
  model = undefined;
  model = await tf.loadLayersModel("cnn_model_e200/model.json");
  console.log("model loaded")
}
loadModel();
