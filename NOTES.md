# Notes

## Assumptions

**Overnight shifts.** A clock-out earlier than the clock-in means the shift crossed midnight. All of them also belong to security guards, so that makes sense.

**Missing clock-outs.** 184 shifts have a clock-in and no clock-out. This is very important for predictions because they are still real shifts. I decided to impute them with that employee's own median shift length.

**The shift hours are capped.** No shift anywhere exceeds 13.5h. 169 are 13.00h, 33 are 13.25h, then **366 at exactly 13.50h**. That looks like a cap.

## How I checked the note sorting and what it found

The note classification is done using 3 different gates. I first labelled each row in `shift_notes.csv` using an LLM. Each note was then classified as nothing_reported, client_requested, disputed_client_request, relief_no_show, colleague_absent, late_handover, equipment_failure or unclear. It first goes through a sequence matcher, which compares it to the 35 most common sentences in the data and classifies it according to that. If the sequence matcher scores a low confidence on any of the notes (meaning that it is a sentence that doesn't look familiar) then it moves to the second gate. This gate basically gets trained on keywords that appear often in the notes. It splits the notes up into chunks and then uses logistic regression to do the actual classification. If this also scores below a certain confidence threshold, it is sent to the last gate, which is where an LLM makes the classification.

To make sure that it actually performs well, I labelled 50 notes myself by hand. They can be found in `/labels/validation_sample.csv`. I could classify Afrikaans and English notes myself, but if a note was in isiZulu I used an LLM to translate it and then labelled it myself. I then tested my classifiers against these hand-labelled notes. The results were that all 50 were correctly labelled. Because the data is synthetic and a lot of sentences are likely repeated, the sequence matcher performed very well, classifying all of the 50 correctly at gate 1. That's why there are 2 fallbacks, because in the real world the notes would differ a lot more.


## The model, and how I would test it properly

For the overtime breach prediction I did train a model. I used ridge regression on only 2 features: the employee's average weekly hours, and hours worked by Wednesday. This sounds simple given all the other data that is available, but it resulted in the best prediction scores. It predicts how many hours that person will work for the rest of the week and then calculates whether they will breach the 10 hour overtime cap.

**What it learned that I didn't expect:** the weight on hours worked so far is negative. One might think a person who has been working a lot will keep working a lot that week, but it is the opposite. They actually decrease their working hours after that. They do, however, work roughly the same amount over each full week.

I tried gradient boosting, empirical-Bayes shrinkage, Poisson counts, quantile regression and a logistic classifier, but these all performed worse than the ridge regression. Adding features also made it worse: 19 features gave a ROC of 0.80 and two gave 0.85. The one thing limiting the performance is the amount of data to train on. Over all the weeks there are only 72 breach events out of more than 1,700 employee-weeks. That is not enough signal for a bigger model to learn from. It does mean, though, that in the real world, if this product gets deployed, it would get better at predicting breaches as more data is fed into it.

**How I would test it without fooling myself:** To test the model I trained on weeks 1 to N and predicted week N+1. That ensures that data isn't leaking and that I'm not testing my model on the data it was trained on. I used precision and recall as my main metrics, and found ridge regression on those 2 features performed best.
