import pandas as pd


def concat_prv_pres(pres, prev_train, prev_test):
    prev = pd.read_csv(prev_train)
    prev['prev'] = True
    prev['diagnosis'] = prev.level
    prev['id_code'] = prev.image.apply(lambda x: f"previous_train_crop/{x}")
    prev_train = prev.drop(['image', 'level'], axis=1)
    prev = pd.read_csv(prev_test)
    prev['prev'] = True
    prev['diagnosis'] = prev.level
    prev['id_code'] = prev.image.apply(lambda x: f"previous_test_crop/{x}")
    prev_test = prev.drop(['image', 'level', 'Usage'], axis=1)
    pres = pd.read_csv(pres)
    pres = pres[pres.diagnosis.notnull()].drop_duplicates(subset='strMd5')
    pres['diagnosis'] = pres.diagnosis.astype(int)
    pres = pres[['id_code', 'diagnosis']].copy()
    pres['prev'] = False
    pres['id_code'] = pres.id_code.apply(lambda x: f"train_crop/{x}")
    df = pd.concat([prev_train, prev_test, pres])
    print(df.head())
    df.to_csv('preprocessed/all.csv', index=False)


concat_prv_pres('preprocessed/strMd5.csv',
                'previous/trainLabels.csv',
                'previous/retinopathy_solution.csv')
