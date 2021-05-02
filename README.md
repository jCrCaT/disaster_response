# Disaster Response Pipeline Project
This is a script that reads tweets from two CSV files (Messages and categories) and predict what kind (36 categories) of help is necessary due to a disaster.

### Installation
To install this script, clone the GitHub repository in your system.
```
git clone Phttps://github.com/jCrCaT/disaster_response.git

```

### Requirements
In order to get all the requirements is necessary to install some packages running the following commands:
```
pip install plotly
pip install pandas
pip install sqlalchemy
pip install sklearn
pip install nltk
pip install flask
```

### Run ETL:
To run the ETL you need to execute the next command on a console:
```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

### Run the model
To run the Script that generates the model you need to execute the following command:
```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

### Execute the server
Run the following command to start the server:
```
python run.py
```

### Interact with the program
Go to http://0.0.0.0:3001/ (in localhost you can use localhost:3001) and type some text related to a disaster response.
