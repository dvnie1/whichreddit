### Setup

Steps:

  1. Create a directory called "model" in the main directory of the app
  2. Paste the trained DistilBERT model into the model directory (config.json, label_mapping.json, model.safetensors, special_tokens_map.json, toknizer_config.json, vocab.txt)
  3. Now, you should be able to run the app with ```flask run```, available on localhost:5000
  4. If you are running the app from the terminal, you might need to export the Flask App variable (instructions below)


Exporting Flask App:

Linux/Mac: ```export FLASK_APP=app.py```
Windows (CMD): ```set FLASK_APP=app.py```
Windows (PowerShell): ```$env:FLASK_APP="app.py"```


Debugging:

If an error occures, make sure that you have installed the requirements ```pip install -r requirements.txt``` and that port 5000 is free.
If you are using port 5000 for something else, you can cahnge the port for the app with adding a tag to the run command like so ```flask run --port=6000```
