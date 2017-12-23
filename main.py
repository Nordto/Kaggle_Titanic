import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler

fileTrain = pd.read_csv('titanic/train.csv')
fileTest = pd.read_csv('titanic/test.csv')
mergeFile = pd.concat([fileTrain, fileTest])

class DataTitanic:
    def __init__(self):
        self.ages = None
        self.fares = None
        self.titles = None
        self.cabines = None
        self.families = None
        self.tickets = None

def getTitle(name):
    if pd.isnull(name):
        return "Null"

    title_search = re.search("([a-zA-Z]+)\.", name)

    if title_search:
        return title_search.group(1).lower()
    else:
        return "None"

def getFamily(row):
    last_name = row["Name"].split(",")[0]

    if last_name:
        family_size = 1 + row["Parch"] + row["SibSp"]

        if family_size > 3:
            return "{0}_{1}".format(last_name.lower(), family_size)
        else:
            return "nofamily"
    else:
        return "unknown"

def getIndex(item, index):
    if pd.isnull(item):
        return -1

    try:
        return index.get_loc(item)
    except KeyError:
        return -1

def overWritingData(data, digest):
    data["AgeRw"] = data.apply(lambda r: digest.ages[r["Sex"]] if pd.isnull(r["Age"]) else r["Age"], axis=1)

    data["FareRw"] = data.apply(lambda r: digest.fares[r["Pclass"]] if pd.isnull(r["Fare"]) else r["Fare"], axis=1)

    genders = { "male": 0, "female": 1 }
    data["SexRw"] = data["Sex"].apply(lambda s: genders.get(s))

    gender_dummies = pd.get_dummies(data["Sex"], prefix = "Sex", dummy_na = False)
    data = pd.concat([data, gender_dummies], axis=1)

    embar = {"C": 0, "S": 1, "U": 2, "Q": 3}
    data["EmbarkedRw"] = data["Embarked"].fillna("U").apply(lambda e: embar.get(e))

    embarkment_dummies = pd.get_dummies(data["Embarked"], prefix = "Embarked", dummy_na = False)
    data = pd.concat([data, embarkment_dummies], axis=1)

    data["RelativesRw"] = data["Parch"] + data["SibSp"]

    data["SingleRw"] = data["RelativesRw"].apply(lambda r: 1 if r == 0 else 0)

    decks = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "T": 7, "U": 8}
    data["DeckRw"] = data["Cabin"].fillna("U").apply(lambda c: decks.get(c[0], -1))

    deck_dummies = pd.get_dummies(data["Cabin"].fillna("U").apply(lambda c: c[0]), prefix="Deck", dummy_na=False)
    data = pd.concat([data, deck_dummies], axis=1)

    title_dummies = pd.get_dummies(data["Name"].apply(lambda n: getTitle(n)), prefix="Title", dummy_na=False)
    data = pd.concat([data, title_dummies], axis=1)

    data["CabinRw"] = data["Cabin"].fillna("unknown").apply(lambda c: getIndex(c, digest.cabines))
    data["TitleRw"] = data["Name"].apply(lambda n: getIndex(getTitle(n), digest.titles))
    data["TicketRw"] = data["Ticket"].apply(lambda t: getIndex(t, digest.tickets))
    data["FamilyRw"] = data.apply(lambda r: getIndex(getFamily(r), digest.families), axis=1)

    return data

if __name__ == "__main__":
    data_tit = DataTitanic()
    data_tit.ages = mergeFile.groupby("Sex")["Age"].median()
    data_tit.fares = mergeFile.groupby("Pclass")["Fare"].median()
    data_tit.titles = pd.Index(fileTest["Name"].apply(getTitle).unique())
    data_tit.families = pd.Index(fileTest.apply(getFamily, axis = 1).unique())
    data_tit.cabines = pd.Index(fileTest["Cabin"].fillna("unknown").unique())
    data_tit.tickets = pd.Index(fileTest["Ticket"].fillna("unknown").unique())


    owTrainFile = overWritingData(fileTrain, data_tit)
    owTestFile = overWritingData(fileTest, data_tit)
    owAllFile = pd.concat([owTestFile, owTrainFile])

    #------
    # print correlation
    #------

    #owAllFile.corr().to_csv("corr.csv", index=False)
    #print(all_data_munged.corr())

    #------

    predictors = ["Pclass",
                  "AgeRw",
                  #"TitleRw",
                  "Title_mr", "Title_mrs", "Title_miss", "Title_master", "Title_ms",
                  "Title_col", "Title_rev", "Title_dr",
                  "CabinRw",
                  #"DeckRw",
                  "Deck_U", "Deck_A", "Deck_B", "Deck_C", "Deck_D", "Deck_E", "Deck_F", "Deck_G",
                  "FamilyRw",
                  "TicketRw",
                  #"SexRw",
                  "Sex_male", "Sex_female",
                  #"EmbarkedRw",
                  "Embarked_S", "Embarked_C", "Embarked_Q",
                  "FareRw",
                  "SibSp",
                  "Parch",
                  "RelativesRw",
                  "SingleRw"]

    cv = StratifiedKFold(fileTrain["Survived"], n_folds = 10, shuffle=True, random_state = 3)

    #------
    # -1..1
    #------

    scaler = StandardScaler()
    scaler.fit(owAllFile[predictors])

    #------

    #--------
    # print predictor
    #--------

    scaledTrainFile= scaler.transform(owTrainFile[predictors])
    scaledTestFile = scaler.transform(owTestFile[predictors])

    selector = SelectKBest(f_classif, k = 5)
    selector.fit(owTrainFile[predictors], owTrainFile["Survived"])

    scores = -np.log10(selector.pvalues_)

    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()

    #--------


    #------
    # output
    #------

    forest = RandomForestClassifier(random_state = 1, n_estimators = 500, min_samples_split = 8, min_samples_leaf = 2)
    scores = cross_val_score(forest, scaledTrainFile, owTrainFile["Survived"], cv = cv, n_jobs = -1)
    print("Random forest values: {}/{}".format(scores.mean(), scores.std()))

    forest.fit(scaledTrainFile, owTrainFile["Survived"])

    predictions = forest.predict(scaledTestFile)

    submission = pd.DataFrame({
        "PassengerId": fileTest["PassengerId"],
        "Survived": predictions
    })

    submission.to_csv("subKaggleTitanic.csv", index=False)

    #------
