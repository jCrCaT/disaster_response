import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import re
 
from nltk import word_tokenize
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import nltk
nltk.download(['punkt', 'wordnet'])

import pickle


def load_data(database_filepath):
    '''
        DEF: Function to load data into dataframe from SQLIte
        INPUT: database_filepath - The path where the file is going to be saved
        OUTPUT:
            X - X Features to feed the model
            y - Y col that specifies the answer to X
            columns.values - The column names of Y 
    '''
    #Connect to the engine
    engine = create_engine('sqlite:///'+database_filepath)
    #Select the dataframe for sqlite
    df = pd.read_sql_query('select * from dataframe',con=engine)
    #Set the features to use in the model (X)
    X = df['message']
    #Set the predictor to use in the model (Y)
    y = df.iloc[:,4:]
    #Convert the value 2 into 1 on the related column
    y.loc[y['related']==2,['related']] = 1
    
    return X,y,y.columns.values


def tokenize(text):
    '''
        DEF: Function used to tokenize the tweets
        INPUT: text - Tweet that is going to be tokenized
        OUTPUT: clean_tokens - Tweets tokenized
    '''
    #Define regex to replace
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    #Find all the strings that interset the url_regex
    detected_urls = re.findall(url_regex, text)
    #For each url detected replace with urlplaceholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    #Call the tokenize method    
    tokens = word_tokenize(text)
    #Lemmatize words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    #For each word lemmatize, lower and strip
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
        DEF: Function that builds the ML model to predict what is the category of the accident that is being succeding
        INPUT: None
        OUTPUT: pipeline: The pipeline that contains the transformation of data (tokenize, tfidf) and the definition of model (RandomForestRegressor)
    '''
    #Define the Pipeline
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('pipe_text',Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())            
        ]))
    ])),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=20)))]
    )
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
        DEF: Function used to evaluate the model
        INPUT:
            model - Model Trained to be evaluated
            X_Test - X Features to evaluate the model
            Y_Test - Y Col that is the response to X
            category_names - 
    '''
    #Run the predictions
    y_pred = model.predict(X_test)
    #Set labels (y_pred uniques)
    labels = np.unique(y_pred)
    #Set the confussion matrix
    confusion_mat = confusion_matrix(Y_test.values.argmax(axis=1), y_pred.argmax(axis=1))
    #Set accuracy if is necessary to show
    accuracy = (y_pred == Y_test).mean()
    #Print the values of predictions with classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
        DEF: Function used to save the model in a .pkl file
        INPUT:
            model - The model (pipeline) that is saved
            model_filepath - The path wh
    '''
    # PKL filename path is the model_filepath variable
    pkl_filename = model_filepath
    #Save the model into PKL file
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    '''
        DEF: 
            Main function used to run the train classifier. What does it do??
            1.- Load data, defining X and y
            2.- Splits the data using train_test_split
            3.- Build the model
            4.- Train the model
            5.- Evaluate the model
            6.- Save the model into file
            
        INPUT: None
        OUTPUT: None
    '''
    if len(sys.argv) == 3:
        #Get the file names
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        #Define X,Y and category names
        X, Y, category_names = load_data(database_filepath)
        #Test split data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        #Buld the model
        model = build_model()
        
        print('Training model...')
        #Train the model
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        #Show metrics
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        #Save the model
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()