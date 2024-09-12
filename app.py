from flask import Flask, render_template, request, jsonify
import pandas as pd
from io import BytesIO
from prep_data import text_handle
from gen_vendor import input_data
from save_excel import create_table_from_df

app = Flask(__name__)

# Function to generate vendor information


@app.route('/')
def index():
    return render_template('index.html', column_name=[])

@app.route('/upload_excel', methods=['POST'])
def upload_excel():
    if 'excel_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['excel_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    
        # Read the file using pandas
    df = pd.read_excel(file)
    print(file)
    # Define the table name
    #table_name = file.filename
    #
    #       ## Get the SQL statement
    #     try:
    #        create_table_from_df(df, table_name)
    #     except Exception as e:
    #         print(e)
           
        
    #     #
    #     # Execute the SQL statement
        
        
        
        
         # Convert the dataframe to a list of dictionaries for JSON response
    data = df.to_dict(orient='records')
    column_names = df.columns.tolist()
    response = {
        'columns': column_names,
        'data': data
    }
    return jsonify(response)
     

@app.route('/generate_vendor', methods=['POST'])
def generate_vendor_route():
    try:
        data = request.json.get('data', [])
       
        l=[]
        for i in data:
            l.append(text_handle(i))
            
        generated=input_data(l)
        print(generated)
        
        result_data = []
        for result_tuple in generated:
            result_data.append({
                'Generated Column 1': result_tuple[0],  # First element of the tuple
                'Generated Column 2': result_tuple[1],   # Second element of the tuple
                'Generated Column 3': result_tuple[2],  # First element of the tuple
                'Generated Column 4': result_tuple[3]   # Second element of the tuple
            })
        

        
         # Return the generated data as JSON
        return jsonify(result_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
