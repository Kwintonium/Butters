# Import data science packages
import pandas as pd
import numpy as np

# Import youtube scraping package
from youtubesearchpython import *

# Neural net
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Plotly
import plotly.express as px
import plotly

def getVideoTitle(video_id):
    ''' This function returns the video title to print on the results screen.
    '''
    video = Video.get(video_id, mode = ResultMode.json, get_upload_date=True)
    return video['title']

def getComments(video_id):
    ''' This function uses the youtubesearchpython API to scrape a YouTube video webpage for its comments.
    '''
    # video_id = "_ZdsmLgCVdU"
    try:
        comments = Comments(video_id)
    except:
        comments = Comments('https://www.youtube.com/watch?v=dQw4w9WgXcQ')

    print(f'Comments Retrieved: {len(comments.comments["result"])}')

    while comments.hasMoreComments:
        print('Getting more comments...')
        comments.getNextComments()
        print(f'Comments Retrieved: {len(comments.comments["result"])}')
        if len(comments.comments["result"])>400: # Change this number to print more results. It is a static number since this is a demo.
            break

    print('Found all the comments.')

    return comments

def getCommentsTable(comments):
    ''' This function serves to strip the comments object and only output the relevant information that is desired on the results table.
    '''

    # Load in comments and votes
    commentsList = []; commentsVotes = []
    commentsTotal = comments.comments.get('result')
    for i in range(len(commentsTotal)):
        commentsVotes.append(commentsTotal[i]['votes'].get('simpleText'))

    # Process the dataframe
    df = pd.DataFrame(commentsTotal) # Convert list of dictionaries to df
    df['votes'] = commentsVotes # Replace votes with uniform vote count
    df.replace(pd.NA, 0, inplace=True) # Replace nans with zeros
    df.replace(np.nan, 0, inplace=True)
    df['replyCount'] = df['replyCount'].astype(int) # Convert float to int
    df['votes'] = df['votes'].replace({'K': '*1e3', 'M': '*1e6'}, regex=True).map(pd.eval).astype(int) # Replace human readable values with ints
    df = df.drop(columns=['author', 'isLiked', 'authorIsChannelOwner', 'voteStatus']) # Drop irrelevant columns to data analytics

    # Replace published with comment age (in months)
    df.insert(3, "commentAge", 0, True) # Create new column
    df['commentAge'] = df['commentAge'].astype('int') # int
    pd.options.mode.chained_assignment = None  # default='warn'
    df['commentAge'][df['published'].str.contains('year')] = df['published'][df['published'].str.contains('year')].str[0:2].astype(int)*12 # multiply year by amount in months
    df['commentAge'][df['published'].str.contains('month')] = df['published'][df['published'].str.contains('month')].str[0:2].astype(int) # months
    df.drop(columns='published', inplace=True) # drop published column
    pd.options.mode.chained_assignment = 'warn'  # default='warn'; turn back on

    return df

def getCommentsSentiment(df):
    ''' This function uses a HugginFace transfer learning model to apply snetiment scores. Operates too slowly for the purposes of this demo, so it is not used.
    '''

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./main/core/sentiment_transfer_learning_tensorflow/')

    # Load model
    loaded_model = TFAutoModelForSequenceClassification.from_pretrained('./main/core/sentiment_transfer_learning_tensorflow/')

    # Tokenize the data
    tokenized_data = tokenizer(df['content'].to_list(), return_tensors='np', padding=True)
    y_predict = loaded_model.predict(dict(tokenized_data), batch_size=128)['logits']

    # Use softmax to find probabilities
    y_probabilities = tf.nn.softmax(y_predict)

    # Predicted Label
    y_class_preds = np.argmax(y_probabilities, axis=1)

    df_preds = pd.Series(y_class_preds, name='labels').map({0: 'Negative', 1: 'Positive'})

    df = pd.concat([df, df_preds], axis=1)

    return df


def getCommentsSentimentVader(df):
    ''' This function uses VADER to retrieve the sentiment scores using a bag of words approach.
    '''

    # Sentiment Score
    sentimentScore = []
    for comment in df['content']:

        sid_obj = SentimentIntensityAnalyzer()
        
        # Sentiment scores are from -1 to 1, so segment them accordingly
        sentiment_dict = sid_obj.polarity_scores(comment)

        if sentiment_dict['compound']>=-0.1 and sentiment_dict['compound']<=0.1:
            sentimentScore.append('Neutral')
        elif sentiment_dict['compound']>=0.1:
            sentimentScore.append('Positive')
        else:
            sentimentScore.append('Negative')
    
    # Store sentiment scores in original dataframe
    df_preds = pd.Series(sentimentScore, name='labels')

    df = pd.concat([df, df_preds], axis=1)

    return df

def pieChart(df):
    ''' This function returns a Plotly pie chart as an HTML object to show on the results webpage.
    '''

    # s = pd.DataFrame(df['labels'])
    color_discrete_map = {'Positive':'green',
                          'Negative':'red',
                          'Neutral':'blue'}
    fig = px.pie(df, values=df['labels'].value_counts().values, names=df['labels'].value_counts().index.values, color=df['labels'].value_counts().index.values, color_discrete_map=color_discrete_map)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_traces(hovertemplate='%{label} <br>%{percent}') #
    fig.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    graph_div = plotly.offline.plot(fig, auto_open = False, output_type="div")
    return graph_div
    

def main(video_id):
   ''' This is the main function that is called in the Django views.py file.
   '''

   video_title = getVideoTitle(video_id)

   comments = getComments(video_id)
   df_comments = getCommentsTable(comments)
#    df_comments = getCommentsSentiment(df_comments)
   df_comments = getCommentsSentimentVader(df_comments)

   graph_div = pieChart(df_comments)

    # sentiment for butters photos
   sentiment = df_comments['labels'].value_counts()['Positive']/len(df_comments.index)
   if sentiment >= 0.6:
        sentiment = 'high'
   elif sentiment < 0.6 and sentiment >= 0.4:
        sentiment = 'medium'
   elif sentiment < 0.4 and sentiment >= 0.05:
        sentiment = 'low'
   else:
        sentiment = 'evil'

   df = df_comments[['content', 'commentAge', 'votes']].loc[df_comments['labels']=='Positive'].sort_values(by=['votes'], ascending=False)
   df.columns = ['Comment', 'Age (Months)', 'Likes']

   df_html = df.to_html(classes='styled-table', index=False)
   df_html = df_html.replace('<th>', '<th align="center" colspan="0">')

   return video_title, df_html, graph_div, sentiment


if __name__ == "__main__":
    # Example
    video_id = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    video_title, df_comments, graph_div = main(video_id)