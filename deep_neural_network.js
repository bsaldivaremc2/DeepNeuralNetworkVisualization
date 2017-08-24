	function sum_matrix(inmatrix,axis=1)
	{
		var tmp_mtrx = inmatrix["_data"];
		var mtrx_size = inmatrix["_size"];
		var mi = mtrx_size[0];
		var mj = mtrx_size[1];
		var tmp_a = [] ;
		for (var i=0;i<mi;i++)
		{
			var sum = 0;
			for (var j=0;j<mj;j++)
			{
				sum+=tmp_mtrx[i][j];
			}
			var cols=[];
			for (var j=0;j<mj;j++)
			{
				cols.push(sum);
			}
			tmp_a.push(cols);
		}
		return math.matrix(tmp_a);
	}

	function initialize_parameters(n_l_l,n_x,n_y,m)
	{
		var n_ll=n_l_l;
		var tmp_params = {};

		for(var li=1;li<n_ll.length;li++)
		{
			n_l = n_ll[li];
			n_l_1 = n_ll[li-1];
			var W_tmp = [];
			for (var i=0;i<n_l;i++)
			{
				var j_tmp = []
				for(var j=0;j<n_l_1;j++)
				{
					j_tmp.push(Math.random());
				}
				W_tmp.push(j_tmp);
			}

			var W = math.matrix(W_tmp);
			var b = math.zeros(n_l,m);
			tmp_params["W"+li]=W;
			tmp_params["b"+li]=b;
		}
		return tmp_params;
	}
	function linear(W,X,b)
	{	
		var wx = math.multiply(W,X);
		var add_b = math.add(wx,b);
		return add_b;
	}
	function activation(z,act_type="sigmoid")
	{
		var tmp_act=0;
		if (act_type=="sigmoid")
		{
			tmp_act = math.dotDivide(1.0,math.add(1.0,math.exp(math.multiply(-1.0,z))));
		}
		else if (act_type=="tanh")
		{
			tmp_act = math.tanh(z);
		}
		return tmp_act;
	}


	function forward_prop(x,params,layers_size,layers_activation)
	{
		var tmp_activations = {}
		var prev_act = x;
		tmp_activations["A"+0]=prev_act;

		for (var li=1;li<layers_size.length;li++)
		{
			var lj=li;
			var tW=params["W"+lj];
			var tb=params["b"+lj];
			var tz = linear(tW,prev_act,tb);
			prev_act = activation(tz,act_type=layers_activation[li-1]);
			tmp_activations["A"+lj] = prev_act;
		}
		return tmp_activations;
	}

	function calc_cost(y,pred)
	{
		var term_y0 = math.dotMultiply(y,math.log(pred));
		var term_y1 = math.dotMultiply(math.subtract(1.0,y),math.log(math.subtract(1.0,pred)));
		return math.multiply(-1.0,math.mean(math.add(term_y0,term_y1)));
	}
	function derivatives(activation,act_type="sigmoid")
	{
		var derivative = 0;
		if (act_type=="sigmoid")
		{
			derivative = math.dotMultiply(activation,math.subtract(1.0,activation));
		}
		else if (act_type=="tanh")
		{
			derivative = math.subtract(1.0,math.dotPow(activation,2));	
		}
		return derivative;
	}
	function back_prop_derivatives(y,pred,activations,params,num_layers,layers_activation)
	{

		var tmp_diff = {};
		var dJ_dAL = math.dotMultiply(-1.0,math.subtract(math.dotDivide(y,pred),math.dotDivide(math.subtract(1.0,y),math.subtract(1.0,pred))))
		
		var dJ_dAprev = dJ_dAL;
		var m = activations["A0"]["_size"][1];
				
		for (var layer_i=num_layers;layer_i>0;layer_i--)
		{
			var layer_activation = activations["A"+layer_i];
			var prev_layer_activation = activations["A"+(layer_i-1)];

			var dA_dZ = derivatives(layer_activation,act_type=layers_activation[layer_i-1]);
			var dJ_dZ = math.dotMultiply(dA_dZ,dJ_dAprev);

			tmp_diff["dW"+layer_i] = math.dotDivide(math.multiply(dJ_dZ,math.transpose(prev_layer_activation)),m);
			tmp_diff["db"+layer_i] = math.dotDivide(sum_matrix(dJ_dZ),m);

			dJ_dAprev = math.multiply(math.transpose(params["W"+layer_i]),dJ_dZ);
		}
		return tmp_diff;
	}
	function update_parameters(params,params_dJ,num_layers,learning_rate=0.1)
	{
		var tmp_dic = {}
		for(var li=1;li<=num_layers;li++)
		{
			tmp_dic["W"+li]=math.subtract(params["W"+li],math.dotMultiply(learning_rate,params_dJ["dW"+li]));
			tmp_dic["b"+li]=math.subtract(params["b"+li],math.dotMultiply(learning_rate,params_dJ["db"+li]));
		}
		return tmp_dic;
	}