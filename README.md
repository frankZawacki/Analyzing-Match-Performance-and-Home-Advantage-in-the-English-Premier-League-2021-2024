# Analyzing-Match-Performance-and-Home-Advantage-in-the-English-Premier-League-2021-2024
# Project Proposal

Michael May, Jessica Vescera, Frank Zawacki, Melina Rodriguez  
DSP 577: Data Enabled Capstone  
Professor Coyne  
March 21, 2026  

---

## Title of the project

Analyzing Match Performance and Home Advantage in the English Premier League (2021–2024)

---

## Aims of the project

The primary aim of this project is to analyze match performance and evaluate the presence of home advantage in the English Premier League using match-level data from the 2021–2024 seasons. This study seeks to identify key factors that influence match outcomes and determine whether variables such as possession, attendance, and team performance metrics contribute to a higher probability of winning. The project will also explore relationships between match statistics and performance outcomes, including whether home teams score more goals than away teams, whether possession percentage is associated with winning probability, and whether higher stadium attendance correlates with improved home team performance. In addition, predictive models will be developed and evaluated to determine whether match statistics can be used to accurately predict match outcomes, while assessing which variables are most important in explaining success in Premier League matches.

To support these aims, the analysis will use the English Premier League Stats (2021–2024) dataset obtained from Kaggle. The data, compiled by Mohamad Sallah from publicly available Sky Sports match statistics, contains 1,140 observations, each representing a single Premier League match played between the 2021 and 2024 seasons. The dataset includes 40 variables describing match characteristics and performance statistics such as teams, goals scored, possession percentages, shooting statistics, disciplinary actions, stadium information, and match attendance. These features include 26 integer variables representing match event counts, 8 numerical variables capturing continuous statistics such as possession and passing, and 6 categorical variables describing match information such as date, stadium, teams, and match outcome, providing a comprehensive foundation for investigating factors that influence match results and home advantage.

---

## Methodologies

Before modeling, we want to acknowledge key strengths and limitations of the dataset and how they affect our analysis. A major strength is that the data are clean and consistent across the 2021–2024 seasons, with rich match-level performance variables (possession, shots, fouls, cards, passing, etc.) plus contextual fields like stadium and attendance. This makes the dataset well-suited for explaining match outcomes and quantifying home advantage through descriptive statistics, visualization, and statistical testing. At the same time, many variables reflect in-game or end-of-match performance, which can create outcome leakage if treated as true predictors (for example, goals are essentially the outcome, and other stats can be influenced by game state). To address this, we will separate the project into (1) an explanatory analysis using the full match statistics and (2) a more conservative predictive analysis that avoids direct-result variables and emphasizes features that are defensible for prediction, such as rolling form features from prior matches, season indicators, and attendance. Finally, attendance may be confounded by team strength and popularity, and the dataset may not include important pre-match factors like injuries, lineups, betting odds, or expected goals. We will mitigate this by controlling for team and season effects where possible and interpreting attendance results cautiously as association rather than causation.

Our tentative plan is built around our main question, what factors influence English Premier League match outcomes, and the supporting questions about home advantage, possession, attendance, and whether possession changes over time.

1. **Data preparation**

   - Cleaning and feature setup  
   - Checking for missing data, duplicates, and inconsistent data  
   - Defining the outcome as home win, draw, or away win, and possibly a simpler binary “home win vs loss” version  
   - Creating difference features, such as:
     - Difference between home shots and away shots  
     - Difference between home shots on target and away shots on target  
     - Fouls difference  
     - Cards difference  
     - Other features that capture who had the edge in a match  

2. **Exploratory Data Analysis (EDA) and statistical checks**

   - Do home teams actually win more often, and by how much (goal difference, win percentage)?  
   - Does higher possession relate to winning, or is the relationship more complex?  
   - Does attendance seem to matter for home performance?  
   - Are there season-to-season changes, such as shifts in possession trends from 2021 through 2024?  
   - Use visualizations (plots, distributions, trend lines) to make the story clear.  

3. **Attendance and home performance**

   When we look at attendance and whether bigger crowds are associated with better home performance, we can define home performance as:

   - Home win rate  
   - Home goals  
   - Goal difference  

   Then we will see how those relate to attendance. Importantly, we will control for factors like team and season when possible, because strong teams might naturally pull bigger crowds, and we do not want to mistake that for a pure attendance effect.

4. **Predictive modeling**

   Finally, to answer the main question—whether match stats can predict winners—we will build a few models:

   - Start with logistic regression for interpretability and a baseline  
   - Compare to models like Random Forest or Gradient Boosting that can capture more complex interactions  
   - Use a season-based split (training on earlier seasons, testing on later seasons) to mimic real-world prediction and avoid learning patterns that would not generalize  
   - Use feature importance to identify which stats matter most for match outcomes and connect these back to home advantage, possession, and attendance  

---

## Possible Outcomes

This project focuses on conducting statistical analysis of match performance and home field advantage in the Premier League from 2021 to 2025. We are going to look at how teams perform in different situations, especially when playing at home versus away, to see if there is a real advantage and how much it actually impacts the result in a real-life game. Along the way in the project, we will create data visualizations to highlight the patterns and trends in match performance, which should make it easier to see how things like possession, shots, and other stats relate to winning or losing in a soccer game. This could help teams in the Premier League know what they should be focusing on to win a game.

We also plan to estimate the probability of different match outcomes (wins, losses, or draws) based on team performance metrics. This will help teams and analysts understand not just what happens in games, but how likely certain outcomes are depending on how a team plays. Overall, the goal is to show how sports data can be used to better understand performance trends and identify competitive advantages within the league that players, teams, or analysts can use in the real world.

We plan to complement the final report with an interactive dashboard, primarily built in Streamlit (Python), and, if time permits, we may also create additional views using RStudio and/or Tableau to support exploration of home advantage, key match statistics, and model-driven outcome probabilities.

On top of this, the insights from this project could be useful for analysts, sports clubs, or researchers who want to build on this kind of work in the future. It can also connect to sports betting, since using real data and identifying trends or probabilities can help make more informed decisions instead of just guessing or going off intuition.

---

## Implications of the proposed project

The implications of this project focus on how analyzing match performance and home field advantage could help to understand what drives success in the English Premier League. By examining how variables such as possession, shots, passing accuracy, fouls, and location relate to match outcomes, the analysis has the potential to show which aspects of performance are strong indicators of success. This also may help to explain how different metrics interact and what they might tell us about winning a match beyond the final score.

Comparing machine learning models may show how various analytical approaches capture patterns and could reinforce the importance of model selection when interpreting performance trends. Clear interpretations of these metrics could support more informed reasoning, such as evaluating how a team usually plays, understanding the influence of home advantage, or considering how quantitative indicators reflect competitive behavior.

Overall, the project might demonstrate how structured analysis of match data can uncover relationships that are not immediately obvious, making it easier to interpret performance in a way that captures underlying dynamics, not just summary statistics alone.

---

## Conclusion

This project will explore how combining different analytical perspectives and professional backgrounds can strengthen the way we look at match performance and home field advantage. Working collaboratively will help us apply our varied experiences to a common problem, improving our skills in data preparation, modeling, and interpretation. This process supports our growth as data scientists and can show how diverse backgrounds can contribute to complex analytical work and more relevant interpretations of real-world data.
