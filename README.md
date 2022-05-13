# This branch is for making edits

---

# Water Wells in Tanzania
![Map of Area](./Images/map.png)

Given a set of data, I need to predict the functionality of pumps into three 
specific classifications:
1. Functional
2. Non-Functional
3. Functional, Needs Repair

# Data
The data provided is from Taarifa and the Tanzanian Ministry of Water. There are
a few things to note about the data.
* 59,400 entries with 40 columns
* The classification for each entry is in a separate csv.
* There are potentially a lot of null values depending on the column we consider.
* Some of the columns provide the same or very similar information.
* After going through the test values for our submission, some of the categorical
  values are not the same and I will have to factor this in at a later point when
  editing.

There is a significant amount of cleaning we need to do. We will also have to
consider which columns to keep depending on the information provided and if
other columns provide the same or similar information.

You can read more about the 
<a href='https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/'>
challenge here.
</a>

# Approach
After cleaning and encoding categorical variables, I will begin building models.
I will make a train-test-split once for cross validation. Then, on my training
set, I will perform another train-test-split. This is so I can try to minimize
any influence our test set would have on our model.
I will make the following models and ensemble methods:
* K-Nearest Neighbors
* Naive Bayes
* Random Forest
* Bagging
* Extra Trees
* XG Boost

I will also use SMOTE to help with balancing classes since there is a huge
imbalance with wells that are classified as needing repair. I will also run
a grid search to hyperparameter tune.

![Class Balance](./Images/class_size.png)

# Current Conclusion
My *best* model appears to be a Random Forest Model after using SMOTE to balance
classes equaly. Here are the current metrics and confusion matrix. My confusion 
matrix includes the precision of our predicted values. A quick breakdown:
* Precision:
  Ratio of our correct guesses to the total guesses for that category.
* Recall:
  Ratio of our correct guesses to the total actual values for that category.
* f1-Score:
  Balance between our precision and recall, and gives a better measure
  of how our model is doing. The closer it is to 1, the better our model does 
  for predicting that classification.
  
For a more in depth explanation of these metrics, go 
<a href='https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/'>
    here
</a>.


|                | precision | recall | f1-score |
|----------------|-----------|--------|----------|
| functional     | 0.81      | 0.87   | 0.84     |
| needs repair   | 0.49      | 0.32   | 0.39     |
| non functional | 0.83      | 0.78   | 0.80     |

![Confusion Matrix](./Images/iamconfusion.png)

The X-axis is what our model predicts, and the Y-axis is what the actual value
would be. The diagonal of this confusion matrix tells us the precision of our
classifications.

Here are the top 10 features my model is using to make classifications.

![Feature Importances](./Images/importances.png)

# Exploratory Data Analysis
Here I explore some of the features my model uses to predict our classifications.

![Classification Map](./Images/map_of_wells.png)

This gives us a quick look of how each classification is spread in Tanzania.
A significant amount of wells that are non-functional are not only in highly
populated areas, but also in areas that are along the borders or in more remote
areas.

![Failure Rate of Well Types](./Images/failurerate.png)

This gives the percentage of wells that are non-functional for each well type.

![Failure Rate of Well Quantities](./Images/quantity.png)

This gives the percentage of wells that are non-functional for each water qunatity
group.

![Top 5 Installers](./Images/func_installer.png)

This gives us the top 5 well installers in Tanzania and the percentege of wells
they made. The Department of Water Engineer(DWE) have made almost 60% of all the
wells in Tanzania.

# Results
I managed to get an f1-score of 0.4191. According to the scoreboard, this ranks 
me as 4965 out of 13541. While the initial score was good, this still shows that 
there is a lot of room for improvement.

# Next Steps
There are several items on my list for improving this project. The first is to 
split the original water_well_work.ipynb notebook into several notebooks and 
source files. This is to split the process into its various components and 
streamline the process into something that would most likely represent an
actual work setting. I will tackle this first to give the structure of this
project new life and to refamilarize myself with the work I had previously done.

After that, I will improve on my data cleaning process to have a more robust 
dataset that may also include newly engineered features. I will then test
different types of machine learning models, including those that I have
learned about since the last time I worked on this project. By the end of my
changes, this repo will look completely different.

# Environment
I am running Python 3.10.4 in a conda environment named "updated". 
I created an environment.yml file for anyone to recreate the environment should
they choose to. <br>
If you were curious on how I exported the environment, I simply did the following:
```
conda activate updated
conda env export > environment.yml
```
If you want to create an environment based on the environment.yml file, do the following:

```
conda env create -f environment.yml
```
</br>