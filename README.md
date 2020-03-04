# Democratic Debate Night Tweets -- Sentiment Analysis

## Project Goals/Overview

With the 2020 Democratic Primaries going on, I thought it would be interesting to use Twitter data and perfomr a sentiment anlysis on tweets involving each candidate in order to see if any trends were clear. 

This project served to:

1. Determine the sentiment of each gathered tweet using a recursive neural network
2. Use these sentiments for in depth analysis of each candidates Twitter sentiment during debates
3. Determine other insights from gathered data
4. Use classifier model to try and identify bots in the data set

The final presentation can be found in the file 'FinalPresentation.pdf' or from this link:
https://www.canva.com/design/DAD0m3zN-cQ/ObBw45vohn39271Mu_56aw/view?utm_content=DAD0m3zN-cQ&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink

## The Data

This project involved gathering data from Twitter, here are the ways and needs of the data gathered:

1. Debate Night Tweets for Sentiment Analysis - Gathered using Python Package GetOldTweets3 to search 'date' & 'serach term' such as Febuary 25 for Joe Biden. These resulted in about 440,000 tweets for 6 different candidates. Twitter only allows a certain number of requests therefore more data was not able to be gathered 'DemDebateTweets.ipynb'.
2. User Data & Tweets for users for Spambot Classifier - Used the Twitter API to gather user data for 20,000 users and around 900,000 tweets 'TwitterAPIcsv/TwitterAPI.ipynb'.

## Sentminet Analysis

For the sentiment analysis I used a pretrained RNN that was trained on movie review data, this can be found in the file 'SentimentAnalysis.ipynb'. The files for the pretrained network can be found from this page: 

## Bot Classification

The Bot classification notbooks can be found in the TwitterAPI folder. In there is the code used to access the twitter API (TwitterAPI.ipynb), and the code where the model was run (TwitterModels.ipynb). 

There is also a file that is used to clean and feature engineer the tweet and user data for the model (dataclean.py).


