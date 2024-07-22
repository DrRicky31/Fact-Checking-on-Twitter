# Fact-Checking for social networks veracity
## Abstract
Online misinformation has become a major concern in recent years and was further emphasized during the COVID-19 pandemic. Social media platforms, such as Twitter, can be serious tools of online disinformation. To better understand the spread of these fake-news, correlations between the following textual characteristics of tweets are analyzed: sentiment, political bias, and truthfulness. Several transformant-based classifiers are trained to detect these textual features and identify potential correlations using conditional label distributions. Our results show how tweets reflect trends in world politics and especially the most misleading ones could have negative impacts on public opinion.
## Code
- `./src/Train-Inference.ipynb` contains some code to train models on the datasets, to use them on other datasets (inference), and to visualise the results into heatmaps. The Load Data cells specify which dataset will be loaded.
- `./src/Train-Inference.py` contains the same code of the python notebook, but it can be executed using CLI Python commands. It also contains code to collect data for inference using SparkStreaming.
- `./src/socketServer.py` contains code to simulate a data streaming environment.
- Datasets should be copied to a new folder named `./data` at the root of the repository. 
- Create a folder in `./data` directory named `models`. This folder will contain the output of the training loop process.
## Accuracy
The execution of the training loop using a training set consisting of three documents belonging to the `Russian-troll-tweets` dataset reports an accuracy of 90% using F1 eval function. For reasons of timing and computing power, all training loop and inference operations were performed on a smaller dataset with a consequent decrease in model performance.
