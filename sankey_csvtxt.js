    //https://developers.google.com/chart/interactive/docs/gallery/sankey

    var default_nodes = [
          [ 'X1', 'L1_1', 5 ],
          [ 'X2', 'L1_1', 7 ],
          [ 'L1_1', 'Y', 6 ],
          [ 'X1', 'L1_2', 2 ],
          [ 'X2', 'L1_2', 9 ],
          [ 'L1_2', 'Y', 4 ]
        ];
    var mod_nodes = [
          [ 'A', 'X', 1 ],
          [ 'B', 'Y', 1 ],
          [ 'B', 'Z', 1 ]
        ];
    var data_array = []
    google.charts.load('current', {'packages':['sankey']});
    google.charts.setOnLoadCallback(drawChart);

      function drawChart(iData=default_nodes) {
        var data = new google.visualization.DataTable();
        data.addColumn('string', 'From');
        data.addColumn('string', 'To');
        data.addColumn('number', 'Weight');
        data.addRows(iData);


        // Sets chart options.
        var options = {
          width: 600,
        };

        // Instantiates and draws our chart, passing in some options.
        var chart = new google.visualization.Sankey(document.getElementById('sankey_basic'));
        chart.draw(data, options);
      }

      $(function(){

        $("#input_csv").change(function(e) 
        {
          data_array = []
          var ext = $("#input_csv").val().split(".").pop().toLowerCase();
          if($.inArray(ext, ["csv"]) == -1) 
          {
            alert('Please Upload CSV');
            return false;
          }    
          if (e.target.files != undefined) 
          {
            var reader = new FileReader();
            reader.onload = function(e) 
            {
              var csvArray=e.target.result.split("\n");
              csvArray.pop();
              //console.log(csvArray);
              var output = '';
              for (i=0;i<csvArray.length;i++) 
              {
                if (csvArray[i])
                {
                  var tmp_row = csvArray[i].split(",");
                  if (tmp_row.length>2)
                  {
                    var num = parseInt(tmp_row[2]);
                  }
                  output+=tmp_row[0]+','+tmp_row[1]+','+num+';';
                }
              }
              $("#input_string").val(output);
            };
            reader.readAsText(e.target.files.item(0));
            e.target.files = undefined;
          }
          return false;
        });

        /*
        $("#proc_data").click(function(){
          console.log("Pressed");

          data_array=[];
          var input_vals = $("#input_string").val();
          var input_splited = input_vals.split(";");
          for (var i=0;i<input_splited.length;i++)
          {
            var row=input_splited[i].split(",");
            if (row.length>2)
            {
              data_array.push([row[0],row[1],parseInt(row[2])]);
            }
          }
          if (data_array.length>1)
          {
            drawChart(data_array);
          }
          else
          {
            alert("No data was loaded");
          }
        });
        */
      });