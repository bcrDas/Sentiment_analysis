# first, we import the relevant modules from the NLTK library
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# next, we initialize VADER so we can use it within our Python script
SIA = SentimentIntensityAnalyzer()

# the variable 'message_text' now contains the text we will analyze.
#message_text = '''Like you, I am getting very frustrated with this process. I am genuinely trying to be as reasonable as possible. I am not trying to "hold up" the deal at the last minute. I'm afraid that I am being asked to take a fairly large leap of faith after this company (I don't mean the two of you -- I mean Enron) has screwed me and the people who work for me.'''
#message_text = '''Looks great.  I think we should have a least 1 or 2 real time traders in Calgary.'''
#message_text = '''I think we are making great progress on the systems side.  I would like to set a deadline of November 10th to have a plan on all North American projects(I'm ok if fundementals groups are excluded) that is signed off on by commercial, Sally's world, and Beth's world.  When I say signed off I mean that I want signitures on a piece of paper that everyone is onside with the plan for each project.  If you don't agree don't sign. If certain projects(ie. the gas plan) are not done yet then lay out a timeframe that the plan will be complete.  I want much more in the way of specifics about objectives and timeframe.Thanks for everyone's hard work on this.'''
#message_text = ''' I am super excited!!!'''

# Continue with the same code the previous section, but replace the *message_text* variable with the new e-mail text:
message_text = '''It seems to me we are in the middle of no man's land with respect to the  following:  Opec production speculation, Mid east crisis and renewed  tensions, US elections and what looks like a slowing economy (?), and no real weather anywhere in the world. I think it would be most prudent to play  the markets from a very flat price position and try to day trade more aggressively. I have no intentions of outguessing Mr. Greenspan, the US. electorate, the Opec ministers and their new important roles, The Israeli and Palestinian leaders, and somewhat importantly, Mother Nature.  Given that, and that we cannot afford to lose any more money, and that Var seems to be a problem, let's be as flat as possible. I'm ok with spread risk  (not front to backs, but commodity spreads). The morning meetings are not inspiring, and I don't have a real feel for  everyone's passion with respect to the markets.  As such, I'd like to ask  John N. to run the morning meetings on Mon. and Wed.  Thanks. Jeff'''

print("\n")
print(message_text)
print("\n")

# Calling the polarity_scores method on sid and passing in the message_text outputs a dictionary with negative, neutral, positive, and compound scores for the input text
sentiment_scores = SIA.polarity_scores(message_text)


# Here we loop through the keys contained in scores (pos, neu, neg, and compound scores) and print the key-value pairs on the screen
for key in sorted(sentiment_scores):
        print('{0}: {1}, '.format(key, sentiment_scores[key]), end='')
        print("\n")
