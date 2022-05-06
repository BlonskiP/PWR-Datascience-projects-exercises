import pandas as pd


class Preprocessor:
    def preprocess(self, data):
        COLUMNS_NOT_NULL = ['location', 'keywords']
        COLUMNS_TO_DROP = ['tweet_id', 'user_id']
        # Delete None location
        preprocessed = data.dropna(subset=COLUMNS_NOT_NULL).reset_index().explode('keywords')
        preprocessed = preprocessed.drop(COLUMNS_TO_DROP, axis=1)
        # Normalize Location jsos
        location_data = pd.json_normalize(preprocessed['location'])
        preprocessed = preprocessed.assign(location=location_data['country'])
        preprocessed = preprocessed.astype({'date': 'datetime64'})
        preprocessed['date'] = preprocessed['date'].dt.normalize()
        # print(preprocessed['keywords'])

        return preprocessed

    def group(self, data):
        preprocessed = data.groupby(['location', 'keywords', 'date'], as_index=False).count()
        return preprocessed

    def preprare_summary(self, df):
        print('summary started')
        cols = ['location', 'text', 'sum', 'date']
        df_text = pd.DataFrame(columns=cols)
        all_locations = df.groupby(['location', 'date'], as_index=False).sum()
        for index, row in all_locations.iterrows():
            date = row['date']
            location = row['location']
            filter = df['location'] == location
            filter2 = df['date'] == date
            data = df.copy().where(filter & filter2).dropna()
            keywords_in_location = data.groupby(['keywords'], as_index=False).sum()
            text = ""
            sum = 0
            for index, row in keywords_in_location.iterrows():
                text += row['keywords'] + " = " + str(row['index']) + " \n "
                sum += row['index']
            df_text = df_text.append(pd.DataFrame([[location, text, sum, date]], columns=cols))
        print('summary done')

        df_text = df_text.sort_values(by=['date'])
        df_text['date'] = df_text['date'].astype('str')
        return df_text
