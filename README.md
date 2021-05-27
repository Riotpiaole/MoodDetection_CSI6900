# Mood Detection 

## Proposal

1. Proposed dataset: 
   - DailyDialog Dataset: emtion distribution diverse
     - The types of emtions:
       - anger 5.87%
       - disgust 2.03%
       - fear 1%
       - joy 74.02%
       - sadness 6.61%
       - suprise 10.47%
       - [Links](https://www.aclweb.org/anthology/I17-1099.pdf)
       - total of 13,118 samples

   - SemEval Sentiment Analysis in Twitter
     - emtions tags are as follows:
       - anger: 23.97%
       - fear: 31.73%
       - joy: 22.7%
       - sadness: 21.6%
       - [links](http://saifmohammad.com/WebPages/TweetEmotionIntensity-dataviz.html)
   - emtion casue:
     - emtion tags are as follows:
       - happiness
       - sadness
       - suprise
       - disgust
       - [links](https://www.site.uottawa.ca/~diana/publications/90420152.pdf)

2. Proposed architecture
   - Bert + classification module
     - testable for classification variation:
       - resnet
       - fully connected layers
