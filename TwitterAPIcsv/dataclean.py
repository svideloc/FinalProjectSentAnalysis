import pandas as pd
import numpy as np

#DATACLEAN OF USER DATA

def clean_data(df_users, df_tweets):
    
    #CLEAN USER DATA
    df_users = df_users.drop(['profile_background_title' ], axis=1)

    df_users['url'] = df_users['url'].isna()
    df_users['url'] = df_users['url'].apply(lambda x: 0 if x == True else 1)

    df_users['profile_banner_url'] = df_users['profile_banner_url'].isna()
    df_users['profile_banner_url'] = df_users['profile_banner_url'].apply(lambda x: 0 if x == True else 1)

    #replaced na's with 0, trues are 1
    df_users['default_profile'] = df_users['default_profile'].fillna(0)
    df_users['geo_enabled'] = df_users['geo_enabled'].fillna(0)

    #replaced na's with 0, trues are 1
    df_users['default_profile'] = df_users['default_profile'].replace(True, 1)
    df_users['default_profile'] = df_users['default_profile'].replace(False, 0)
    df_users['geo_enabled'] = df_users['geo_enabled'].replace(True, 1)
    df_users['geo_enabled'] = df_users['geo_enabled'].replace(False, 0)

    df_users['profile_background_image_url_https'] = df_users['profile_background_image_url_https'].fillna(df_users['profile_background_image_url_https'].mode()[0])
    df_users['profile_background_image_url'] = df_users['profile_background_image_url'].fillna(df_users['profile_background_image_url'].mode()[0])

    #This for loop loops through the below columns and makes them binary, either ARE YOU UNIQUE == 1, if not == 0

    cols_to_binary = ['profile_background_image_url_https','profile_text_color','profile_sidebar_border_color','profile_sidebar_fill_color',
                     'profile_background_image_url', 'profile_background_color', 'profile_link_color']
    for col in cols_to_binary:
        gp = df_users[col].value_counts().to_frame().reset_index()
        gp.columns = [col, 'counts']
        df_users = df_users.merge(gp, on=col, how='left')

        #Set threshold and adjust orignal column, drop merged column
        df_users[col] = np.where(df_users['counts']==1, 1, 0)
        df_users = df_users.drop(['counts'], axis=1)
    
    # CLEAN TWEET DATA
    df_tweets = df_tweets.drop(['favorited', 'retweeted', 'contributors', 'place', 'geo', 'truncated'], axis=1)
    
    df_tweets['favorite_count'] = df_tweets['favorite_count'].replace('False', 0)
    df_tweets['favorite_count'] = df_tweets['favorite_count'].fillna(0)
    df_tweets["favorite_count"] = pd.to_numeric(df_tweets["favorite_count"])

    #in_reply_to_user_id grouping by if the count is greater than 5000 or not
    gp = df_tweets.in_reply_to_user_id.value_counts().to_frame().reset_index()
    gp.columns = ['in_reply_to_user_id', 'counts']
    df_tweets = df_tweets.merge(gp, on='in_reply_to_user_id', how='left')

    #Set threshold and adjust orignal column, drop merged column
    df_tweets['in_reply_to_user_id'] = np.where(df_tweets['counts']>5000, 0, 1)
    df_tweets = df_tweets.drop(['counts'], axis=1)

    # String to be searched in start of string  
    search ="RT"
    search2 ="@"

    # boolean series returned 
    df_tweets['retweet_y_n'] = df_tweets["text"].str.startswith(search) 
    df_tweets['contains_@'] = df_tweets["text"].str.contains(search2)

    #Replace Bool with int
    df_tweets['retweet_y_n'] = df_tweets['retweet_y_n'].replace(True, 1) 
    df_tweets['retweet_y_n'] = df_tweets['retweet_y_n'].replace(False, 0) 
    df_tweets['contains_@'] = df_tweets['contains_@'].replace(True, 1) 
    df_tweets['contains_@'] = df_tweets['contains_@'].replace(False, 0) 

    cols = ['user_id', 'retweet_count', 'favorite_count']

    df1 = df_tweets[cols].groupby('user_id').mean() #retweet & favorite count mean for each user
    df2 = df_tweets[cols].groupby('user_id').max() #retweet & favorite count max for each user
    df3 = df_tweets[cols].groupby('user_id').min() #retweet & favorite count min for each user
    df4 = df_tweets[cols].groupby('user_id').std() #retweet & favorite count std for each user
    df5 = df_tweets.groupby(['user_id'])['source'].agg(pd.Series.mode).to_frame() #source mode for each user
    df6 = df_tweets.groupby('user_id')['in_reply_to_user_id'].mean() #in_reply_to_any_user_id percentage of tweets
    df7 = df_tweets.groupby('user_id')['retweet_y_n'].mean() #retweet percentage
    df8 = df_tweets.groupby('user_id')['contains_@'].mean() #@ someone percentage

    df_features = pd.merge(df1,df2,on='user_id')
    df_features = pd.merge(df_features, df3, on= 'user_id')
    df_features = pd.merge(df_features, df4, on= 'user_id')
    df_features = pd.merge(df_features, df5, on='user_id')
    df_features = pd.merge(df_features, df6, on='user_id')
    df_features = pd.merge(df_features, df7, on='user_id')
    df_features = pd.merge(df_features, df8, on='user_id')

    #new df column names
    cols = ['retweet_mean', 'favorite_count_mean', 'retweet_max', 'favorite_max', 'retweet_min', 'favorite_min', 'retweet_std', 'favorite_std', 'source_mode', 'in_reply_user_id_mean', 'retweet_%', 'contains_@_%']

    df_features.columns = cols
    df_features = df_features.reset_index()

    df_features['source_mode'] = df_features.source_mode.astype(str)

    #in_reply_to_user_id grouping by if the count is greater than 5000 or not
    gp = df_features.source_mode.value_counts().to_frame().reset_index()
    gp.columns = ['source_mode', 'counts']
    df_features = df_features.merge(gp, on='source_mode', how='left')

    #Set threshold and adjust orignal column, drop merged column
    df_features['source_mode'] = np.where(df_features['counts']<100, '1', df_features['source_mode'])
    df_features = df_features.drop(['counts'], axis=1)

    df_features['source_mode'] = df_features.source_mode.replace('<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>', 'Twitter Web App')
    df_features['source_mode'] = df_features.source_mode.replace('1', 'Other')
    df_features['source_mode'] = df_features.source_mode.replace('<a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>', 'Twitter for iPhone')
    df_features['source_mode'] = df_features.source_mode.replace('<a href="http://twitter.com/download/android" rel="nofollow">Twitter for Android</a>', 'Twitter for Android')
    
    #MERGE DATAFRAMES
    df_final = df_features.merge(df_users, left_on='user_id', right_on='id')
    
    cols = ['retweet_mean', 'favorite_count_mean', 'retweet_max', 'favorite_max', 'retweet_std', 'favorite_std', 'in_reply_user_id_mean', 'retweet_%', 'contains_@_%', 'default_profile', 'geo_enabled']
    for c in cols:
        df_final[c] = df_final[c].astype(float)
        
    cols = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'favourites_count', 'listed_count', 'url', 'profile_banner_url', 'profile_background_image_url_https', 'profile_text_color', 'profile_sidebar_border_color', 'profile_sidebar_fill_color', 'profile_background_image_url', 'profile_background_color', 'profile_link_color']
    
    for c in cols:
        df_final[c] = df_final[c].astype(int)
    
    return df_final