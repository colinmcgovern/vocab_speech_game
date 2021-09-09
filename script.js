function argMax(array) {
	return array.indexOf(Math.max(...array));
}

function load(id,model){
	
	image = document.getElementById(id);
	tensor = tf.browser.fromPixels(image,1)
			.resizeNearestNeighbor([64, 64]).toFloat();
	tensor = tf.expandDims(tensor, axis=0)

	let start = performance.now();
	prediction = model.predict(tensor).data();
	prediction.then(function (p){
		console.log(id,argMax(p),performance.now()-start);
	});	
}

async function run() {
    const model = tf.loadLayersModel('exported_model/model.json');

    model.then(function (res){


    	// for (let j = 0; j < 100; j++) { 
	    // 	for (let i = 0; i < 10; i++) { 
	    // 		load(i.toString(),res)
	    // 	}
	    // }

	    load(0,res)

    });
    
};

run();
