# Project 6 - Capstone Project
## NBAPlayerAnalyzer App: Recommending App built to provide statistically similar players  
#### Carlos B Rivera
#### *DSIR22221-E Capstone, Presented 5.14.21*

# Executive Summary
Prior to the arrival of players like Kevin Durant, most players height dictated what role they would play on a basketball team. If a player was above 6'10 he would be considered a center and was assigned a specific role they rarely stepped out of. The modern NBA players that don't adopt multiple roles have to fates: either really good at their specifc role, or they fade out of the league within a few years. The topic of "positionless basketball" is a common discussion on major sports talk shows like ESPN and FoxSports. This project attempts to reclassify players into modern categories based on how their stats are similar. 

Using a recommender system built on cosine similarities to reclassify players is the goal of this project. Different scalars will be used to train KMeans models and the best scoring models will be used to create clusters for EDA and player classification. 

Given the feature rich dataset, I separated the features based on how well they described one of four categories: defense, offense, traditional, and pure offense performance. The baseline model was trained with standard scalar and 5 clusters. Baseline scores were low as expected: offensive category(sil = 0.17495), defensive category(sil = 0.1742), pure_offense category(sil = 0.20842), and traditional category(sil =0.20837). Next a manual gridsearch was conducted to find the best n_clusters with the highest silhouette(sil) score. KMean models again were trained using three different scalars (MinMax, Normalizer,Standard Scalar) and 2-5 n_clusters. These baseline models did much better than the baseline models with the following scores: offensive category(sil = 0.730029, 4 clusters), defensive category(sil = 0.36, 2 clusters), pure_offense category(sil = 0.320131, 3 clusters), and traditional category(sil =0.453440, 3 clusters). Further modeling was done by using PCA reduction and then training KMean models for all categories. These models also did much better than the baseline models with the folowing scores: offensive category(sil =  0.723362, 3 clusters, 4 components), defensive category(sil = 0.361631, 2 clusters, 3 components), pure_offense category(sil = 0.315019, 3 clusters, 3 components), and traditional category(sil =0.474175, 4 clusters, 5 components). After careful consideration I chose to go with a mix of models that best represented what I believed would be an accurate representation of modern play styles.

After the modeling process EDA was done on each winning models clusters to investigate and successfully reclassify players by modern playing styles. However, for stronger proof of these clusters I created a recommender system that provides the top ten similar players of any 2021 NBA player. The results were as we suspected with recommendations being soley on skills rather than height. Moving forward, the goal of this project is to train different models like DBSCAN and Hierarchal clustering. Also enhance the online tool to also give recommendations on incoming rookie prospects.  

# Mission Statement
**The NBA has been going through a revolutionary period in which players no longer fit traditional positional roles.** Evidently, every position on the basketball court has adopted and evolved into new roles with more responsibilities. This project not only **aims to reclassify traditional roles, it aims to create a recommender system to demonstrate that the current league is indeed.**
# Datasets
### Data Colleciton
This project used data scraped from NBA.com (https://stats.nba.com/players/advanced/?sort=PTS&dir=-1) and basketball-reference.com (https://www.basketball-reference.com/leagues/NBA_{}_per_game.html).

During the scrape from the basketball-reference website from years 2000-2019 a dataset of 28 features with 11576 rows. The header was scraped multiple times, so those needed to be dropped. In addition to dropping multiple headers, rows with missing data were also dropped since they accounted for less than 10% of the data. The dataset contains players more than once just at different ages in their career. This was useful with regard to tracking the evolution of traditional players and their roles.

During the second scrape from the NBA.com website I attained all of the offensive, defensive, traditional, and advanced statistics from the current 2021 season. This dataset was absent of missing values and any header issues. This dataset was used during the clustering stage to reclassify players into modern roles. Final dataset contained 84 features with 534 players(rows). 

### Data Dictionary

|Feature	|Type	|Dataset	|Description|
|-	|-	|-	|-	|
|Player	|	object	|trad_stats/nba_stats2021_original|Player name	|
|Pos	|	object	|trad_stats|Position Player Plays	|
|Age/AGE	|	Int64	|trad_stats/nba_stats2021_original|Player age	|
|Tm/Team	|	object	|trad_stats/nba_stats2021_original|	Current player team-	|
|G/GP	|int64	|trad_stats/nba_stats2021_original	|Games played during season	|
|GS|int64	|trad_stats	|Number of Games player started	|
|MP/MIN	|float64|trad_stats/nba_stats2021_original	|Average playing time per game	|
|FG/FGM	|float64	|trad_stats/nba_stats2021_original	|Number of field goal makes per game	|
|FGA	|float64	|trad_stats/nba_stats2021_original	|Number of field goal attempts per game	|
|FG%	|float64	|trad_stats/nba_stats2021_original	|% field goal average (FGM/FGA)	|
|3P/3PM	|float64	|trad_stats/nba_stats2021_original	|Number of 3 pointers made per game	|
|3PA	|float64	|trad_stats/nba_stats2021_original	|Number of 3 pointers attempted per game	|
|3P%	|float64	|trad_stats/nba_stats2021_original	|% three point goal average (3PM/3PA)	|
|%FGA3PT	|float64	|nba_stats2021_original	| % of 3 pointer field goals attempted by a player	|
|3FGM%AST	|float64	|nba_stats2021_original	| % of 3 pointer scored by a player that were assisted	|
|3FGM%UAST	|float64	|nba_stats2021_original	| % of 3 pointer scored by a player that were unassisted	|
|2P/2PM	|float64	|trad_stats/nba_stats2021_original	|Number of 2 pointers made per game	|
|2PA	|float64	|trad_stats/nba_stats2021_original	|Number of 2 pointers attempted per game	|
|2P%	|float64	|trad_stats/nba_stats2021_original	|% two point goal average (2PM/2PA)	|
|%FGA2PT	|float64	|nba_stats2021_original	| % of 2 pointer field goals attempted by a player	|
|%PTSPITP	|float64	|nba_stats2021_original	| % of 2 pointer(points in the paint) scored by a player	|
|eFG%	|float64	|trad_stats/nba_stats2021_original	|% Effective field Goal or how well you score overall	|
|TS%	|float64	|nba_stats2021_original	|True Shooting %	|
|FT/FTM	|float64	|trad_stats/nba_stats2021_original	|Number of free-throws( 1 pointers) made per game	|
|FTA	|float64	|trad_stats/nba_stats2021_original	|Number free-throws( 1 pointers) attempted per game	|
|FT%	|float64	|trad_stats/nba_stats2021_original	|% free throw( 1 pointers) average (FTM/FTA)	|
|ORB/OREB	|float64	|trad_stats/nba_stats2021_original	|Offensive rebounds per game	|
|DRB/DREB_x	|float64	|trad_stats/nba_stats2021_original	|Defensive rebounds per game	|
|TRB	|float64	|trad_stats/nba_stats2021_original	|Total rebounds (ORB+DRB)	|
|OREB%	|float64	|nba_stats2021_original	|% of offensive rebounds a player is responsible for	|
|DREB%_x	|float64	|nba_stats2021_original	|% of defensive rebounds a player is responsible for	|
|REB%	|float64	|nba_stats2021_original	|% of available rebounds a player grabbed while on the floor	|
|AST	|float64	|trad_stats/nba_stats2021_original	|Assist per game	|
|AST%	|float64	|nba_stats2021_original	|% of assist player has when in the game	|
|AST/TO	|float64	|nba_stats2021_original	|Assist to turnover ratio	|
|AST Ratio	|float64	|nba_stats2021_original	|Number of assists a player averages per 100 possessions	|
|2FGM%AST	|float64	|nba_stats2021_original	| % of 2 pointer scored by a player that were assisted	|
|2FGM%UAST	|float64	|nba_stats2021_original	| % of 2 pointer scored by a player that were unassisted	|
|2FGM%AST	|float64	|nba_stats2021_original	| % of total field goals made by a player that were assisted	|
|2FGM%UAST	|float64	|nba_stats2021_original	| % of total field goals made by a player that were unassisted	|
|STL/STL_x	|float64	|trad_stats/nba_stats2021_original	|Number of steals per game	|
|BLK/BLK_x	|float64	|trad_stats/nba_stats2021_original	|Number of blocks per game	|
|TOV	|float64	|trad_stats/nba_stats2021_original	|Turnovers per game	|
|PF	|float64	|trad_stats/nba_stats2021_original	|Personal Fouls per game	|
|PTS	|float64	|trad_stats/nba_stats2021_original	|Points per game	|
|%PTS2PT	|float64	|nba_stats2021_original	| % of 2 pointer scored by a player	|
|%PTS2PT MR	|float64	|nba_stats2021_original	| % of 2 pointer (mid-range) scored by a player	|
|%PTSFBPs	|float64	|nba_stats2021_original	| % of 2 pointer(fast break points) scored by a player	|
|%PTSFT	|float64	|nba_stats2021_original	| % of free-throws(1 point) scored by a player	|
|%PTSOffTO	|float64	|nba_stats2021_original	| % of generated by turnovers	|
|%PTSPITP	|float64	|nba_stats2021_original	| % of 2 pointer(points in the paint) scored by a player	|
|DD2	|float64	|nba_stats2021_original	|Double Doubles (Number of games where a player accumulates a double-digit total in two of five statistical categories (PTS, AST, REB, BLKs, OREB))	|
|TD3	|float64	|nba_stats2021_original	|Triple Doubles (Number of games where a player accumulates a double-digit total in three of five statistical categories (PTS, AST, REB, BLKs, OREB))	|
|FP	|float64	|nba_stats2021_original	|Fantasy points	|
|+/-	|float64	|nba_stats2021_original	|Plus-minus score( how well the team performs with the player on the floor)	|
|OFFRTG	|float64	|nba_stats2021_original	|Offensive Rating(Measures how many points are scored per 100 possesions while player is on the floor)	|
|DEFRTG	|float64	|nba_stats2021_original	|Defensive Rating(Measures how many points allowed per 100 possesions while player is on the floor)	|
|NETRTG	|float64	|nba_stats2021_original	|Offensive Rating( Measures a team's point differential per 100 possessions while player is on court)	|
|DEFWS	|float64	|nba_stats2021_original	|Share of wins a player contributes to their team from defense	|
|USG%	|float64	|nba_stats2021_original	|Usage Percentages ( % of team plays used by a player when he is on the floor)	|
|PACE	|float64	|nba_stats2021_original	| % number of possessions per 48 minutes for a team or player	|
|PIE	|float64	|nba_stats2021_original	| % Player Impact Estimates (measures a player's overall statistical contribution against the total statistics in games they play in)	|
|POSS	|int64	|nba_stats2021_original	| The number of possessions played by a player	|
|OPP PTSOFF TOV	|float64	|nba_stats2021_original	| Opponent Points off Turnovers commited by player	|
|OPP PTS2ND CHANCE	|float64	|nba_stats2021_original	| The number of points an opposing player or team scores on offensive rebounds	|
|OPP PTSFB	|float64	|nba_stats2021_original	| Opponent points during fastbreak	|
|OPP PTSPAINT	|float64	|nba_stats2021_original	| Opponent points in the paint	|
|%FGM	|float64	|nba_stats2021_original	| % of a teams field goals made contributed by a player while on the court	|
|%FGA	|float64	|nba_stats2021_original	| % of a teams field goals attempted contributed by a player while on the court	|
|%3PM	|float64	|nba_stats2021_original	| % of a teams 3 pointer field goals made contributed by a player while on the court	|
|%3PA	|float64	|nba_stats2021_original	| % of a teams 3 pointer field goals attempted contributed by a player while on the court	|
|%FTM	|float64	|nba_stats2021_original	| % of a teams free-throws made contributed by a player while on the court	|
|%FTA	|float64	|nba_stats2021_original	| % of a teams free-throws attempted contributed by a player while on the court	|
|%OREB	|float64	|nba_stats2021_original	| % of a teams offensive rebounds contributed by a player while on the court	|
|%DREB	|float64	|nba_stats2021_original	| % of a teams defensive rebounds contributed by a player while on the court	|
|%REB	|float64	|nba_stats2021_original	| % of a teams rebounds contributed by a player while on the court	|
|%AST	|float64	|nba_stats2021_original	| % of a teams assist contributed by a player while on the court	|
|%TOV	|float64	|nba_stats2021_original	| % of a teams turnovers contributed by a player while on the court	|
|%STL	|float64	|nba_stats2021_original	| % of a teams steals contributed by a player while on the court	|
|%BLKA	|float64	|nba_stats2021_original	| % of a teams attempted blocks contributed by a player while on the court	|
|%BLK	|float64	|nba_stats2021_original	| % of a teams blocks contributed by a player while on the court	|
|%PF	|float64	|nba_stats2021_original	| % of a teams personal fouls contributed by a player while on the court	|
|%PFD	|float64	|nba_stats2021_original	| % of a team's personal fouls drawn that a player has while on the court	|
|%PTS	|float64	|nba_stats2021_original	| % of a team's personal fouls drawn that a player has while on the court	|



## Analysis

## Baseline KMean Model Build and Analysis
### Data Cleaning Up/ Preprocessing (NBA.com)
- A for loop was used to change all the dtypes of all columns to either object, float64, int64.
- Used apply function to remove the commas from positions (POSS) to be able to train models.
- During the scrape of all 4 category stats (defensive, offensive, advanced, traditional) there were no missing data points. 
- Dropped duplicate features.

### Modeling, Iteration, & Evaluation of Baseline KMean Model
Given the rich list of features in the dataset, features were separated into individual list based on the category of statistics. 

> **Offensive features**: AGE, PTS, 3PM, 3PA, 3P%, AST, TOV, AST%, AST/TO, AST\xa0Ratio, TO\xa0Ratio, eFG%, POSS, %PTS2PT, %PTS2PT\xa0MR, %PTS3PT, %PTSFBPs, %PTSFT, %PTSOffTO, %PTSPITP, 2FGM%AST, 2FGM%UAST, 3FGM%AST, 3FGM%UAST, %PF, %PFD, %PTS

> **Defensive features**: DREB_x, REB, STL_x, BLK_x, DD2, DEFRTG, DREB%_x, REB%, DEF\xa0RTG, %DREB_x, STL%, %BLK_x, DEFWS, %STL, %BLKA 

> **Pure Offense**: 3P%, AST%, eFG%, %PTS2PT, %PTSOffTO, %PTSFT, %PTSFBPs, %PTSPITP, 2FGM%AST, 3FGM%AST, 2FGM%UAST

> **Traditional features** MIN, FGM, FGA, FG%, 3PM, 3PA, 3P%, FTM, FTA, FT%, OREB, DREB_x, REB, AST, STL_x, BLK_x, PF, PTS, +/-

       **Results/Evalutation**
        
            Offensive Category:
                - n_clusters: 5
                - silhouette score: 0.1742
                - inertia score: 4416.8
                
            Defensive Category:
                - n_clusters: 5
                - silhouette score: 0.17495
                - inertia score: 8706.8
                
            Traditional Category:
                - n_clusters: 5
                - silhouette score: 0.20842
                - inertia score: 2904.2
                
            Pure Offense Category:
                - n_clusters: 5
                - silhouette score: 0.208372
                - inertia score: 4322.9
                
---

### Baseline KMean Model Build
- For the baseline model standard scalar was used to scale the data. 
- 5 centriods were used since there are 5 tradtional positions in the NBA. 
- The models for each category were expected to perform poorly with low silhouette scores and large inertia scores. This is due to the initial belief that the NBA positions have evolved and accpeted more roles. 

## KMean Manual GridSearch Model

### Modeling, Iteration, & Evaluation of KMean Manual GridSearch Model
- Feature list from baseline model were used for the modeling process. All models scored better than the baseline model with relatively decent number of clusters with respect to the category.  

       **Results/Evalutation**
        
            Offensive Category:
                - Best K clusters = 4
                - Best scalar = Normalize
                - Sil score = 0.730029
                
            Defensive Category:
                - Best K clusters = 2
                - Best scalar = Normalize
                - Sil score = 0.36
                
            Traditional Category:
                - Best K clusters = 3
                - Best scalar = Normalize
                - Sil score = 0.453440
                
            Pure Offense Category:
                - Best K clusters = 3
                - Best scalar = Normalize
                - Sil score = 0.320131
    


### KMean Manual GridSearch Model Build
- A function was built to manually gridsearch through the number of n_clusters and through 3 different scalars.
- Scalars used were MinMax, Normalizer, Standard Scalar
- Range of n_clusters used were 2-10 

## KMean Manual GridSearch Model w/ PCA
### Modeling, Iteration, & Evaluation of KMean Manual GridSearch Model w/ PCA
Feature list from baseline model were used for the modeling process. Scoring better than the baseline was expected since the initial idea was that players no longer fit into traditional roles. However, models like Offensive category did really well scoring much higher than baseline.

            Offensive Category:
                - n_components = 4 (88% explained variance)
                - sil score = 0.723362
                - clusters = 3 (different types of players)
                
            Defensive Category:
                - n_components = 3 (84% explained variance)
                - sil score = 0.361631
                - clusters = 2 (different types of players)
                
            Traditional Category:
                - n_components = 5 (97% explained variance)
                - sil score = 0.474175
                - clusters = 4 (different types of players)

            Pure Offense Category:
                - n_components = 3 (84% explained variance)
                - sil score = 0.315019
                - clusters = 3 (different types of players

### KMean Manual GridSearch Model w/ PCA Build
- Given the feature rich clusters of the dataset, feature reduction seemed like the right move. 
- As expected the KMean with PCA models all performed better than the baseline models. 
## Winning Models
Since not one model build had all the best scores, a combination of models were choosen to reclassify players into modern categories. 


            Offensive Category:
                - Best K clusters = 4
                - Best scalar = Normalize
                - Sil score = 0.730029
                
            Pure Offense Category:
                - Best K clusters = 3
                - Best scalar = Normalize
                - Sil score = 0.320131

            Traditional Category:
                - n_components = 5 (97% explained variance)
                - sil score = 0.474175
                - clusters = 4 (different types of players)
                
            Defensive Category:
                - n_components = 3 (84% explained variance)
                - sil score = 0.361631
                - clusters = 2 (different types of players)

### Winning Models EDA/Reclassification of players

**Offensive Category** 
>**cluster0(Practice Players)**: 
These are your typical practice players.
     
     - Accountable for the most of the personal fouls while there on the court. (Refs have a bias towards these players since they believe they are the less skilled players on the team)
     - Barely any possesions 
     - Ex. Anderson Varejao, Noah Vonleh
>**cluster1(SuperStar)**: These players are generally your official starters and team superstars.

      - Score better when assisted
      - They typically have low assist since they either create scoring opportunities for themselves and teammates depend on them for scoring. 
      - Ex. Steph Curry, Luka Doncic, Bradley Beal

>**cluster2(WorkHorse)**: These players are typically your energy players off the bench.

       - Most of the points come from hustle plays
       - Are responsible for defending the other teams best player
       - 27% of fouls while their on the court are commited by this player

>**cluster3(Diverse-Tall)**: These players are considered your third option scores.

       - Similar to the impact player, to a lesser degree.
       - Responsible for 51% of the 2pts scored when on the floor.
       - Ex. TJ Warren, Jabari Parker 
**Defensive Category**
>**cluster0(Unicorns)**: These players literally are a cheat code.

    - Besides being a complex problem on the offensive end of the court, they still play above average defense.
    - Responsible for 16% of steals, 45% of the blocks, and 28% of the rebounds when on the court. 
    - Ex. Kevin Durant, Giannis Antetokounmpo
>**cluster1(1st Ballot HOF)**: First ballet HOF who are also skilled and well rounded at defending the opposing teams best player.
    
    - Only difference between Unicorns and 1st Ballot HOF is their height.
    - Defend shorter players, but also do everything else very well.
    - Responsible for 20% of the teams steals when on the court. 
    - Ex. Damian Lillard, Bradley Beal
**Traditional Category**
>**cluster 0(corner Stones)**: These players can be labeled as Impact Players. 
    
    - Cornerstone players.
    - Responsible for most of the scoring and motivation
    - Ex. Kawhi Leonard
    
>**cluster 1(Defender/Intentional foul)**: These players once in awhile have big plays due to their risky playing style. 

    - Low plaiyng time, but when played defense is their assignment
    - Players in this cluster have an average +/- of -6.9 when on the floor. 
    - Ex. Noah Vonleh
    
>**cluster 2(Star Protector)**: Players in this cluster typically are known for defending their best players and/or intimidating the opposing best player
    
    - These players get just enough time in the game create tension. Not always effective.
    - Ex. Jared Dudley, Jordan Bell

>**cluster 3(Defensive Anchors)**: Players known for controlling who scores in the paint.

    - Typically impact the game on the defensive end making sure there is minimum scoring near the basket
    - Offensively play a ton of pick and roll
    - Indirectly effects the development of the offense. 
    - Ex. Rudy Gobert, Enes Kanter
**Pure Offense Category**
>**cluster0(Creator)**: These are players your defensive scheme is not built around, but good enough that you must know their location at all times. 
    
    - They can slash and take you off the dribble to score, or collapse the defense to create opportunites for others. 
    - Responsible for 70% of the teams unassisted 2 pointers. 
    - Ex. Luka Donic, Ben Simmons
    
>**cluster1(Heat Checkers)**: These Players will continue to score if they get the sense that their hot. 

    - Responsible for 94% of a teams assisted 3's
    - 53% effective field goal average
    - Ex. Thomas Bryant

>**cluster2(Driver)**: Players that tend to drive and score the basketball within the arc. 
    
    - These players are very skilled at scoring within the arc, and occasionally can make 3 pointers.
    - Responsible for 65% of the points scored in the paint, and 70% of a teams 2 point shots/layups. 
    - Ex. Iman Shumpert, Giannis Antetokounmpo

#### NBAPlayerAnalyzer App
The app is simple to use and provides fast recommendations. Since the models were trained on 2021 NBA stats, it only takes a 2021 player. Once provided a player name, the app provides the top 10 similar players based on the desired category (offensive, defensive, pure offense, traditional).The app was built using Streamlit app to provide passionate fans and professional teams an app that quickly finds similar players based on the input player. Teams can now better plan and strategize for specific players by finding similar players that defensively or offensively match up well. 
To infer back on the mission statement, the recommender app also offers compelling evidence for our initial theory that the NBA is positionless. To highlight one case, recommendations for player Anthony Davis produced player Jarret Culver as a recommendation. Anthony Davis is a 6'10ft player with small forward skills, while Jarret Culver in terms of height is a true small forward. This evidence suggest that a players height are becoming less important in defining their traditional roles. It is evidently all about skill level and how many traiditonal roles can you play.  
### Conclusions & Future Directions
The goal of reclassifying players into traditional roles highlighted what we initially thought, the NBA is heading towards a complete positionless league where a players hieght no longer dictates what his skill set is. All of the category models performed better than the baseline models, again proving that players can be clustered in less than 5 roles. The NBAPlayerAnalyzer app highlights the top 10 similar players when provided with a 2021 NBA players name. For different recommendations, all you have to do is change the category to the right. 
Future plans for the app include building models on data from current NBA players rookie statistics to provide insight on incoming prospects. Questions like the following: How does prospect X compare to rookie LeBron James? or What is prospect X's cieling? Floor?, will be able to be answered. In addition to enhancing the app, training different models like DBSCAN or Hierarchical Clustering models could produce some interesting results. The main evaluation metric used in this project was the silhouette score, since the statistical relevance to one another is essential during player reclassification. 

