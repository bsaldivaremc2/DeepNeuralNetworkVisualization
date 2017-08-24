

$(function(){

	$("#proc_data").click(function(){

		$("#stop").click(function(){
				current_iter=num_iters;
		});		

		var X=[];
		var X_in = ($("#X").val()).split(";");
		for(xi=0;xi<X_in.length;xi++)
		{
			X.push((X_in[xi].split(",")).map(Number));
		}
		X = math.matrix(X);
		var Y=[];
		var Y_in = ($("#Y").val()).split(";");
		for(yi=0;yi<Y_in.length;yi++)
		{
			Y.push((Y_in[yi].split(",")).map(Number));
		}
		Y = math.matrix(Y);
	
/*	
  	var X = math.matrix([[2, 4, 6, 8, 1], 
  		                 [1, 2, 7, 6 , 2]]);


  	var Y = math.matrix([
  		[1, 1, 0, 0, 1]
  		]);
*/
  		X=math.dotDivide(X,math.sum(X));
		n_x=X["_size"][0];
		m=X["_size"][1];
		n_y=Y["_size"][0];


		var learning_rate=parseFloat($("#learning_rate").val());
		var current_iter=0
		var num_iters = parseInt($("#num_iters").val());
		var iter_bool=true;
		var layers_size=(($("#layers_size").val()).split(",")).map(Number);

		var layers_activation_type = $("#act_function").val();
		var update_visual_neurons_count = parseInt($("#update_visual_neurons_count").val());; //"n" interations
		var update_iter_time = parseInt($("#update_iter_time").val());;//milliseconds

/*
	var learning_rate=1;
	var current_iter=0
	var num_iters = 500;
	var iter_bool=true;
	var layers_size=[2,2,1];
	var layers_activation_type = "tanh";
	var update_visual_neurons_count = 4; //"n" interations
	var update_iter_time = 100;//milliseconds

*/
		layers_size.splice(0,0,n_x);
		layers_size.push(n_y);
		var layers_activation =	[];
		for (var lt=0;lt<layers_size-2;lt++)
		{
			layers_activation.push(layers_activation_type);
		}
		layers_activation.push("sigmoid");

		
		var num_layers=layers_size.length-1;
		var params = initialize_parameters(layers_size,n_x,n_y,m); 
		var activations = forward_prop(X,params,layers_size,layers_activation);
		var AL = activations["A"+num_layers];
		var back_for_var = setInterval(backward_forward, update_iter_time);

		var cost = calc_cost(Y,AL);
		$("#init_cost").html("Initial cost:  "+cost);
		var params_dJ = {};
		var visual_neurons = []

				function backward_forward()
		{
			if (current_iter<num_iters)
			{
				params_dJ = back_prop_derivatives(Y,AL,activations,params,num_layers,layers_activation);
				params = update_parameters(params,params_dJ,num_layers,learning_rate);
				activations = forward_prop(X,params,layers_size,layers_activation);
				AL = activations["A"+num_layers];
				cost = calc_cost(Y,AL);
				var predictions = math.round(AL);
				$("#cost").html("Cost at iter: "+current_iter+" is: "+cost);
				$("#truth").html("Target binary classes: "+Y["_data"]);
				$("#prediction").html("prediction: "+predictions["_data"]);
				visual_neurons = update_visual_neurons();
				if (current_iter%update_visual_neurons_count==0)
				{
					console.log("Update");
					drawChart(iData=visual_neurons);	
				}
				current_iter+=1;
			}
		}

		function update_visual_neurons()
		{
			var neuron_pairs_edges = [];
			for(var li=1;li<layers_size.length;li++)
			{
				var first_l = "X";
				var next_l = "Y";		
				for( var ln=1;ln<=layers_size[li-1];ln++)
				{				
					if ((li-1)==0)
					{
						first_l="X_"+ln;
					}
					else
					{
						first_l="L"+(li-1)+"_"+ln;
					}
					for( var lnn=1;lnn<=layers_size[li];lnn++)
					{						
						if (li==(layers_size.length-1))
						{
							next_l="Y_"+lnn;
						}
						else
						{
							next_l="L"+(li)+"_"+lnn;
						}
						var lw = params["W"+li]["_data"][lnn-1][ln-1];
						lw = Math.abs(Math.ceil(lw*100+1));
						neuron_pairs_edges.push([first_l,next_l,lw]);
					}
				}
			}		
			return neuron_pairs_edges;
		}

	});
});
