# BigData: Fact-Checking for social networks veracity
## Abstract
Online misinformation has become a significant issue in recent years, especially highlighted during the COVID-19 pandemic. Social media platforms like Twitter often act as major channels for spreading misinformation. To gain a deeper understanding of how false information, lies, deceptions, and rumors proliferate, we examine the relationships between various textual features in tweets, such as emotion, sentiment, political bias, stance, veracity, and conspiracy theories. We use multiple datasets to train several transformer-based classifiers to detect these textual features and identify potential correlations by analyzing the conditional distributions of the labels.
## Code
- `./src/Train-Inference.ipynb` contains some code to train models on the datasets, to use them on other datasets (inference), and to visualise the results into heatmaps. The Load Data cells specify which dataset will be loaded.
- `./src/Train-Inference.py` contains the same code of the python notebook, but it can be executed using CLI Python commands.
- Datasets should be copied to a new folder named `./data` at the root of the repository. 
- Create a folder in `./data` directory named `models`. This folder will contain the output of the training loop process.
## Accuracy
The execution of the training loop using a training set consisting of three documents belonging to the `Russian-troll-tweets` dataset reports an accuracy of 90% using F1 eval function. For reasons of timing and computing power, all training loop and inference operations were performed on a smaller dataset with a consequent decrease in model performance.
