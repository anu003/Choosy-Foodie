# Choosy-Foodie
A restaurant recommendation system based on user reviews.

As opposed to the traditional restaurant recommendation systems, which recommends based on overall average ratings of a restaurant irrespective of users preferences, Choosy Foodie extracts the aspects and the associated opinions of a restaurant reviewed by each user and then recommends a restaurant based on users preferences.

Each review of a user is split into sentences, on which parts-of-speech tagging is applied and nouns are extracted as features and verbs, adverbs, adjectives as opinion words. In order to group the features into aspects of a restaurant, topic modelling has been performed and the extracted topics were binned into four categories food, service, ambience and price.

Further, sentiment analysis is performed on each sentence of a review and a score was calculated from the polarities obtained. This score was normalized to fall into a range of 0 to 5, which represents the rating of each aspect of a restaurant reviewed by a user.  

Finally, concern, requirement and satisfaction was calculated to obtained for each user-restaurant pair and for a given user the restaurants with top satisfaction ratings is recommended to the user.



