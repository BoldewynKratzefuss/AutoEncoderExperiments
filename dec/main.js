async function run() {
   try {
     // create a new session and load the AlexNet model.
     const session = await ort.InferenceSession.create('./decoder.onnx');
 
     // prepare dummy input data
     const dims = [1,2];
     const size = dims[0] * dims[1];
     const inputData = Float32Array.from({ length: size }, () => Math.random());
     console.log(inputData);
 
     // prepare feeds. use model input names as keys.
     const feeds = { input_2: new ort.Tensor('float32', inputData, dims) };
 
     // feed inputs and run
     const results = await session.run(feeds);
     const imageData = await results.conv2d_transpose_2.toImageData();

     const canvas = document.getElementById("canvas");
     var bob = canvas.getContext("2d");
     var bobImg = new Image();
     bobImg.addEventListener("load", function() {
        bob.drawImage(imageData, 1, 1, 40, 40);
     }, false);
     bobImg.src = "moo.png";
     
     return results;
   } catch (e) {
     console.log(e);
   }
 }

 
 
run();


