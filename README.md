# exploratory_data_analysis_demonstration


The Financial Phrasebank dataset from Hugging Face was selected for EDA. The reason that this dataset was chosen is as follow:

1- This data can help to gain better understanding of the financial phrase data by exploring its characteristics, such as the distribution of sentiment scores, the frequency of words or phrases, and the relationships between different variables.

2- The reliability of the labels assigned by annotators can be examined 

The steps of EDA on the dataset:

1- Inspection of the data 
		Print the first five rows of the dataset.
		Check the sentiment label distribution.
		
2- Check for missing data

3- Calculate sentiment polarity
		Values are in the range of [-1,1] where 1 means positive sentiment and -1 means a negative sentiment.
		Plot the histogram of the polarity
		It can be seen from the histogram that most of the sentences polarity scores are close to zero (neutral).		
		
4- Compare the label and polarity distribution when polarity is categorized in three bins 
		It shows that polarity is most close to neutral compare to labels.
		
5- Compare the effect of the threshold on polarity distribution with label.
		By changing the threshold we can see how much the distribution can be similar to the label.
		
6- Check the effect of a specific word on label
		It can be seen that by considering specific word shuch as 'increase', the label, (tonality), can be changed and predicted.
		
7- Get the sentence lengths and plot the distribution of sentence lengths.
		It can be seen from the histograms that neutral sentences have bigger lenght compare to positive and negative sentences. 
		
# Run the EDA code:

eda_financial_phrasebank.ipynb

The notebook file file is provided to do the exploratory task.
The .py file can be used if you need it. 
 
