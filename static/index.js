// https://github.com/bhattbhavesh91/tf-js-example-1
// https://machinelearningknowledge.ai/learn-image-classification-with-tensorflow-js-using-mobilenet-model-web-app
// https://towardsdatascience.com/deploying-an-image-classifier-using-javascript-84da1480b3a4
var img;
$(document).ready(function() {
     $('#loading').hide();
     $('select').on('change', function() {
        document.getElementById("imageDiv").style.display = "block";
        document.getElementById("selected-image").src = this.value;
        $("#prediction-list-div").empty();
        $('#loading').hide();
        img = new Image()
        img.src=document.getElementById("selected-image").src;
        img.crossOrigin = "anonymous";
        
    });
    $('#predict-button').on('click', function() {
        getPred(img);
    });
    
});
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
    $('#loading').show();
    document.getElementById("imageDiv").style.display = "block";
    document.getElementById("field_value").value = "0";    
    var oFReader = new FileReader();
    oFReader.readAsDataURL(document.getElementById("image-selector").files[0]);
    oFReader.onload = function (oFREvent) {
        document.getElementById("selected-image").src = oFREvent.target.result;
        $("#prediction-list-div").empty();
        $('#loading').hide();
        img = $("#selected-image").get(0);
    };
};

async function getPred(img){
    $('#loading').show();
	// Read the image and perform preprocessing
   
    let image = img;
    console.log(image)
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
	

    colorList = ["bg-success", "bg-info", "bg-warning", "bg", "bg-danger"]
   
    for(i=0;i<top5.length;i++){
        const probability = (top5[i].probability * 100).toFixed(2);
        $("#prediction-list-div").append("<div class='row'><div class='col-4'>" + top5[i].className + "</div><div class='progress' style='width: 50%;'><div class='progress-bar " + colorList[i] + "' role='progressbar' style='width: " + probability + "%'>" + probability + "%</div></div></div><br>")
        if(i==top5.length-1){
            $('#loading').hide();
        }
    }
}

async function loadModel() {
  model = undefined;
  model = await tf.loadLayersModel("https://raw.githubusercontent.com/MonicaRaj/KvasirClassifier/main/static/cnn_model_e200_64x64/model.json");
  console.log("model loaded")
}
loadModel();
